import argparse
import logging
import os
from pathlib import Path
from traceback import print_exc

from openai import OpenAI

# from openagi_main import openagi_main as openagi_gpt_main
# from mixtral_main import mixtral_main as openagi_mixtral_main
from main import main as openagi_main
from utils.flow_utils import set_logger, ReadLineFromFile, get_response_from_client

import random

import torch

import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from tqdm import tqdm

from flow.flow import Flow

from openagi_api.combine_model_seq import SeqCombine
from openagi_api.general_dataset import GeneralDataset
from utils.agi_utils import match_module_seq, txt_eval, image_similarity
from evaluate import load
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore

from undecorated import undecorated
from utils.finetune_utils import construct_optimizer
from types import MethodType
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, DistributedType
from peft import PeftModel, PeftModelForCausalLM, prepare_model_for_int8_training, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, AutoFeatureExtractor, MixtralForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

def autoagi_gpt(args):
    task_description_list = ReadLineFromFile("../openagi_data/task_description.txt")
    task_description = task_description_list[0]

    autoagi_instruction = f'You are a proficient expert in designing workflows for complex task planning and can revise existing workflows based on their execution performances.\n' \
                          f'Example task descriptions:\n```{task_description}```\n```{task_description_list[-1]}```\n\n' \
                          f'An execution large language model will receive the task description as query, and then follow your generated workflow for providing plans as the task solution.\n' \
                          f'You must provide a workflow each time, and the User will reply to you with the performances from the execution large language model. ' \
                          f'The execution performance is a float number between 0 and 1; the higher, the better. '

    user_instruction = f'Provide a workflow with several steps. Each step is a one-line string. ' \
                       f'Each step is in the form of: ```[Step Name]:::[Step Type]:::[Step Instruction]:::[Step Branch]```\n' \
                       f'[Step Type] could be "process", "terminal", or "decision".\n' \
                       f'[Step Branch] consists of several branches. Each branch is in the form of "[Key]::[Branch Step Name]" and separated by "::".\n' \
                       f'Note: "process" step has exactly one branch, with "next" as the key; "decision" step has more than one branches; ' \
                       f'"terminal" step has zero branch, indicating the end of working flow, but there could be multiple "terminal" steps.\n' \
                       f'At least one "terminal" step exists, meaning the last step of the workflow!\n' \
                       f'[Key], [Step Name], and [Step Instruction] are all in the string form.\n' \
                       f'[Branch Step Name] should be appear as a unique [Step Name] in the workflow.\n' \

    chat_history = [{'role': 'system', 'content': autoagi_instruction}, {'role': 'user', 'content': user_instruction}]
    manual_flow = '\n'.join(ReadLineFromFile(args.flow_file))
    chat_history.append({'role': 'assistant', 'content': manual_flow})
    args.dataset = 'train'
    baseline = openagi_main(args)
    args.dataset = 'test'
    reward = openagi_main(args)
    logging.info(f'```\nReward:\n{reward}```\n')
    chat_history.append({'role': 'user', 'content': f'The execution performance of given workflow is {baseline}. '
                                                    f'Provide a new workflow in the same form of previous one.'})

    openai_key = args.openai_key
    client = OpenAI(api_key=openai_key)
    args.flow_file = args.auto_flow_file

    for epoch in range(args.auto_epochs):
        res = get_response_from_client(client, chat_history, args.auto_model_name)[0]
        fout = open(args.flow_file, 'w')
        fout.write(res)
        fout.close()
        chat_history.append({'role': 'assistant', 'content': res})
        logging.info(f'```\nFlows:\n{res}```\n')
        try:
            args.dataset = 'train'
            reward = openagi_main(args)
            chat_history.append({'role': 'user', 'content': f'The execution performance of given workflow is {reward}. '
                                                            f'Provide a new workflow in the same form of previous one.'})
            logging.info(f'```\nReward:\n{reward}```\n')
            if reward > baseline:
                baseline = reward
                logging.info(f'\n\nNew Testing:\n\n')
                args.dataset = 'test'
                reward = openagi_main(args)
                logging.info(f'```\nTesting Reward:\n{reward}```\n')
                args.dataset = 'train'

        except Exception as e:
            print_exc()
            chat_history.append({'role': 'user', 'content': f'When executing the workflow, there is an error: {e}. \n'
                                                            f'Please re-generate a new workflow in the same form of the first one.'})
            logging.info(f'```\nError:\n{e}```\n')

def grammar_check(client, flow):
    user_instruction = f'Here is the requirement of a workflow with several steps:\nEach step is a one-line string. ' \
                       f'Each step is in the form of: ```[Step Name]:::[Step Type]:::[Step Instruction]:::[Step Branch]```\n' \
                       f'[Step Type] could be "process", "terminal", or "decision".\n' \
                       f'[Step Branch] consists of several branches. Each branch is in the form of "[Key]::[Branch Step Name]" and separated by "::".\n' \
                       f'Note: "process" step has exactly one branch, with "next" as the key; "decision" step has more than one branches; ' \
                       f'"terminal" step has zero branch, indicating the end of working flow, but there could be multiple "terminal" steps.\n' \
                       f'At least one "terminal" step exists, meaning the last step of the workflow!\n' \
                       f'[Key], [Step Name], and [Step Instruction] are all in the string form.\n' \
                       f'[Branch Step Name] should be appear as a unique [Step Name] in the workflow.\n' \
                       f'Please answer whether the following workflow satisfy all the requirements. Only answer "Yes" or "No". Do not be verbose!'

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": user_instruction},
            {"role": "user", "content": f'{flow}'}
        ],
        temperature=0.,
    )

    response = response.choices[0].message.content

    logging.info(f'Grammar Check:\n{response}')

    if 'yes' in response.lower():
        return flow

    user_instruction = f'Parser the following result to a workflow with several steps. Each step is a one-line string. ' \
                       f'Each step is in the form of: ```[Step Name]:::[Step Type]:::[Step Instruction]:::[Step Branch]```\n' \
                       f'[Step Type] could be "process", "terminal", or "decision".\n' \
                       f'[Step Branch] consists of several branches. Each branch is in the form of "[Key]::[Branch Step Name]" and separated by "::".\n' \
                       f'Note: "process" step has exactly one branch, with "next" as the key; "decision" step has more than one branches; ' \
                       f'"terminal" step has zero branch, indicating the end of working flow, but there could be multiple "terminal" steps.\n' \
                       f'At least one "terminal" step exists, meaning the last step of the workflow!\n' \
                       f'[Key], [Step Name], and [Step Instruction] are all in the string form.\n' \
                       f'[Branch Step Name] should be appear as a unique [Step Name] in the workflow.\n' \
                       f'Avoid empty line!\n' \
                       f'Keep the original information as much as possible.\n'

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": user_instruction},
            {"role": "user", "content": f'{flow}'}
        ],
        temperature=0.,
    )

    response = response.choices[0].message.content

    print(f'Corrected Flow :\n{response}')
    return response

def autoagi_mixtral(args):
    tokenizer = AutoTokenizer.from_pretrained(args.auto_model_name, cache_dir=args.cache_dir,
                                              use_fast=True,)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.auto_model_name, cache_dir=args.cache_dir,
                                                              load_in_4bit=True,
                                                              device_map="auto",
                                                              peft_config=lora_config, )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    optimizer, scheduler = construct_optimizer(args, model, 20)

    task_description_list = ReadLineFromFile("../openagi_data/task_description.txt")
    task_description = task_description_list[0]

    user_instruction = f'You are a proficient expert in designing workflows for complex task planning. Provide a workflow with several steps. Each step is a one-line string. ' \
                       f'Each step is in the form of: ```[Step Name]:::[Step Type]:::[Step Instruction]:::[Step Branch]```\n' \
                       f'[Step Type] could be "process", "terminal", or "decision".\n' \
                       f'[Step Branch] consists of several branches. Each branch is in the form of "[Key]::[Branch Step Name]" and separated by "::".\n' \
                       f'Note: "process" step has exactly one branch, with "next" as the key; "decision" step has more than one branches; ' \
                       f'"terminal" step has zero branch, indicating the end of the workflow, but there could be multiple "terminal" steps.\n' \
                       f'At least one "terminal" step exists, meaning the last step of the workflow!\n' \
                       f'[Key], [Step Name], and [Step Instruction] are all in the string form.\n' \
                       f'[Branch Step Name] should be appear as a unique [Step Name] in the workflow.\n'

    manual_flow = '\n'.join(ReadLineFromFile(args.flow_file))

    question = f'Provide a workflow in the same form of the example workflow, so that the workflow can be executed ' \
               f'to solve the complex planning tasks like the example ones when the execution model are only given ' \
               f'task description. No Explanation! Do not be verbose! The answer should be a workflow in the same form of the Example workflow!'

    prompt = f'### Instruction:\n{user_instruction}\n' \
             f'### Example task description:\n{task_description}\n{task_description_list[-1]}\n\n### Example workflow:\n{manual_flow}\n\n### Question:\n{question}\n\n### Answer:\n'

    print(prompt)

    config = PPOConfig(
        model_name=args.auto_model_name,
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
    )

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        optimizer=optimizer,
    )

    openai_key = args.openai_key
    client = OpenAI(api_key=openai_key)
    args.flow_file = args.auto_flow_file
    baseline = -1.0

    for epoch in range(args.auto_epochs):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        generation_kwargs = dict(temperature=1.0, do_sample=True, top_p=0.9, top_k=40, max_new_tokens=1024)

        output = ppo_trainer.generate(torch.squeeze(input_ids), return_prompt=False, **generation_kwargs)

        out_flow = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        logging.info(f'```\nRaw Generated Flows:\n{out_flow}```\n')

        out_flow = grammar_check(client, out_flow)

        fout = open(args.flow_file, 'w')
        fout.write(out_flow)
        fout.close()
        logging.info(f'```\nFlows:\n{out_flow}```\n')

        try:
            args.dataset = 'train'
            reward = openagi_main(args)

        except Exception as e:
            print_exc()
            logging.info(f'```\nError:\n{e}```\n')
            reward = -2.0
            # exit(0)

        logging.info(f'```\nReward:\n{reward}```\n')
        if reward > baseline:
            baseline = reward
            logging.info(f'\n\nNew Testing:\n\n')
            args.dataset = 'test'
            reward = openagi_main(args)
            logging.info(f'```\nTesting Reward:\n{reward}```\n')
            args.dataset = 'train'
            Path(args.output_dir + f"/model/step_{epoch}").mkdir(parents=True, exist_ok=True)
            Path(args.output_dir + f"/ppo_trainer/step_{epoch}").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir + f"/model/step_{epoch}")
            ppo_trainer.save_pretrained(args.output_dir + f"/ppo_trainer/step_{epoch}")

        reward = [torch.tensor(reward)]
        # print(reward, input_ids[0], output[0])
        train_stat = ppo_trainer.step([input_ids[0]], [output[0]], reward)
        logging.info(f'\n\nFinish step {epoch}!!!!!\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", type=str, default='')
    parser.add_argument("--claude_key", type=str, default='')
    parser.add_argument("--hf_token", type=str, default='')
    parser.add_argument("--cache_dir", type=str, default='../cache_dir/')
    parser.add_argument("--log_file_name", type=str, default='../log/autoagi.txt')
    parser.add_argument("--log_dir", type=str, default='')
    # parser.add_argument("--tool_file", type=str, default='./info/OpenAGI/tools.txt')
    parser.add_argument("--tool_name", type=str, default='tools.txt')
    # parser.add_argument("--flow_file", type=str, default='./info/OpenAGI/OpenAGI_Flow.txt')
    parser.add_argument("--flow_name", type=str, default='OpenAGI_Flow.txt')
    parser.add_argument("--auto_flow_name", type=str, default='auto_OpenAGI_Flow.txt')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto_epochs", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_round", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='test')
    parser.add_argument("--auto_model_name", type=str, default='gpt-4-1106-preview', help='Flow Generator LLM name.')
    parser.add_argument("--model_name", type=str, default='gpt-4-1106-preview', help='Execution LLM name.')
    parser.add_argument("--task", type=str, default='OpenAGI')
    parser.add_argument("--other_info_name", type=str, default='other_info.txt')
    parser.add_argument("--max_fail_times", type=int, default=2, help='Max allow fail times on tools arg choice')
    parser.add_argument("--get_observation", type=str, default='traverse',
                        help='How to get observations, "traverse" stands for asking one by one, "direct" stands for directly asking.')
    parser.add_argument("--info_dir", type=str, default='./info/')

    parser.add_argument("--set_type", type=str, default='validation')
    parser.add_argument("--results_dir", type=str, default='../results/')
    parser.add_argument("--results_name", type=str, default='sample')

    parser.add_argument("--num_seq", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default='./')

    args = parser.parse_known_args()[0]
    args.device_list = ["cuda:0", "cpu"]
    args.huggingface_cache = args.cache_dir
    args.log_name = os.path.join(args.log_dir, args.log_file_name)
    set_logger(args)

    args.flow_file = os.path.join(args.info_dir, args.task, args.flow_name)
    args.auto_flow_file = os.path.join(args.info_dir, args.task, args.auto_flow_name)
    args.tool_file = os.path.join(args.info_dir, args.task, args.tool_name)
    args.other_file = os.path.join(args.info_dir, args.task, args.other_info_name)

    if 'tral' in args.auto_model_name:
        autoagi_mixtral(args)
    elif 'gpt' in args.auto_model_name:
        autoagi_gpt(args)
    else:
        raise NotImplementedError