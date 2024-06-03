import argparse
import logging
import os

import numpy as np
from datasets import load_dataset

from tqdm import tqdm

from sentence_transformers import SentenceTransformer


from utils.notebook import Notebook
from utils.flow_utils import ReadLineFromFile, get_prompt, get_observation, notebook_summarize, get_response_from_client, check_tool_use, check_tool_name, \
    get_tool_arg, check_branch, set_logger
from flow.flow import Flow
from openai import OpenAI

from openagi_api.combine_model_seq import SeqCombine
from openagi_api.general_dataset import GeneralDataset
from utils.agi_utils import match_module_seq, txt_eval, image_similarity, parse_module_list_with_gpt
from evaluate import load
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore


from vllm import LLM, SamplingParams
import torch
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoFeatureExtractor



# from evaluation import OpenAGI_evaluate


def global_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", type=str, default='')
    parser.add_argument("--claude_key", type=str, default='')
    parser.add_argument("--model_name", type=str, default='gpt-4-1106-preview')
    parser.add_argument("--cache_dir", type=str, default='../cache_dir/')
    parser.add_argument("--task", type=str, default='TravelPlanner')
    parser.add_argument("--data_dir", type=str, default='../travel_database/')
    parser.add_argument("--info_dir", type=str, default='./info/')
    parser.add_argument("--results_dir", type=str, default='../results/')
    parser.add_argument("--results_name", type=str, default='sample')
    parser.add_argument("--flow_name", type=str, default='TravelPlanner_flight_Flow.txt')
    parser.add_argument("--tool_name", type=str, default='tools.txt')
    parser.add_argument("--other_info_name", type=str, default='other_info.txt')
    parser.add_argument("--log_dir", type=str, default='../log/')
    parser.add_argument("--set_type", type=str, default='validation')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--get_observation", type=str, default='traverse', help='How to get observations, "traverse" stands for asking one by one, "direct" stands for directly asking.')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_fail_times", type=int, default=2, help='Max allow fail times on tools arg choice')
    parser.add_argument("--max_round", type=int, default=100, help='Max allow round of executions')
    parser.add_argument("--log_file_name", type=str, default='travelplanner.txt')
    args = parser.parse_known_args()[0]
    return args

def finish_one_task(instruction, tool_info, other_info, flow, task_idx, query, tool_list, notebook, args, client):


    notebook.reset()

    plan_round = 1
    flow_ptr = flow.header
    logging.info(f'```\ntask id:\n{task_idx}```\n')
    logging.info(f'```\nquery:\n{query}```\n')

    total_price = 0.0
    
    current_progress = []
    questions, answers, output_record = [], [], []  # record each round: question, LLM output, tool output (if exists else LLM output)
    tool_calling_list = []
    return_res = dict()
    while True:
        if plan_round >= args.max_round:
            break
        chat_history = []
        if isinstance(instruction, str):
            chat_history.append({
                'role': 'system',
                'content': instruction
            })

        # First determine whether need information in notebook
        observations, observation_summary, price = get_observation(client, query, current_progress, notebook, flow_ptr, args.get_observation, model_name=args.model_name)
        total_price += price

        # generate prompt
        prompt = get_prompt(tool_info, flow_ptr, query, current_progress, observations, args.model_name, other_info)
        logging.info(f'Prompt: \n```\n{prompt}\n```')
        chat_history.append({
            'role': 'user',
            'content': prompt
        })
        # get response from LLM
        res, price = get_response_from_client(client, chat_history, model_name=args.model_name)
        res = res.replace('```', '"')
        total_price += price
        logging.info(f'Response: \n```\n{res}\n```')
        chat_history.append({
            'role': 'assistant',
            'content': res
        })
        questions.append(str(flow_ptr))
        answers.append(str(res))
        current_progress.append(f'Question {plan_round}: ```{flow_ptr.get_instruction()}```')
        # current_progress.append(f'Answer {plan_round}: ```{res}```')

        # check tool use
        if len(tool_list) > 0:
            try:
                tool_use, price = check_tool_use(client, '\n'.join(tool_calling_list), flow_ptr, str(res), tool_info, model_name=args.model_name)
                total_price += price
            except Exception as e:
                logging.error(f"Error when checking tool use: {e}")
                tool_use = False
            
            if tool_use:
                tool_name, price= check_tool_name(client, flow_ptr, str(res), list(tool_list.keys()), model_name=args.model_name)
                total_price += price
                tool = tool_list[tool_name]
                for k in range(args.max_fail_times):
                    try:
                        param, price = get_tool_arg(client, flow_ptr, str(res), tool_info, tool_name, model_name=args.model_name)
                        total_price += price
                        if param == 'None':
                            tool_result = tool.run()
                        else:
                            param_sep = [p.strip() for p in param.strip().split(',')]
                            tool_result = tool.run(*param_sep)
                        tool_calling = f'{tool_name} [ {param} ]'
                        if tool_calling in tool_calling_list:
                            current_progress.append(f'Answer {plan_round}: ```{res}```')
                            break
                        tool_calling_list.append(tool_calling)
                        short_summary, price = notebook_summarize(client, tool_info, tool_calling, args.model_name)
                        total_price += price
                        msg = notebook.write(f'Round {plan_round}', tool_result, short_summary)
                        logging.info(f"Save the observation into notebook: {msg}")
                        current_progress.append(f'Answer {plan_round}: Calling tool ```{tool_calling}```. Short Summary: {short_summary}.')
                        break
                    except Exception as e:
                        logging.error(f"Error when getting tool arguments: {e}")
                        if k + 1 == args.max_fail_times:  # Max Fail attempts
                            logging.error('Reach Max fail attempts on Get Tool Parameters.')
                            # if reach max fail attempts, do not use tool in this step.
                            current_progress.append(f'Answer {plan_round}: ```{res}```')
                            break
                            # exit(1)
                        else:
                            continue
            else:
                current_progress.append(f'Answer {plan_round}: ```{res}```')
                # output_record.append(None)

        # terminate condition
        if len(flow_ptr.branch) == 0 and flow_ptr.type.lower() == 'terminal':
            if args.task == 'OpenAGI':
                eval_device = "cuda:0"
                clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
                vit_ckpt = "nateraw/vit-base-beans"
                vit = AutoModel.from_pretrained(vit_ckpt)
                vit.eval()
                vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt)

                f = transforms.ToPILImage()
                bertscore = load("bertscore")

                data_path = "../openagi_data/"
                dataset = GeneralDataset(task_idx, data_path)
                dataloader = DataLoader(dataset, batch_size=args.batch_size)
                seq_com = SeqCombine(args)
                module_list = parse_module_list_with_gpt(args, res).split(',')
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
                module_list = match_module_seq(module_list, sentence_model).split(',')
                print(module_list)
                seq_com.construct_module_tree(module_list)
                task_rewards = []
                for batch in tqdm(dataloader):
                    inputs = [list(input_data) for input_data in batch['input']]
                    try:
                        predictions = seq_com.run_module_tree(module_list, inputs, dataset.input_file_type)
                        if 0 <= task_idx <= 14:
                            outputs = list(batch['output'][0])
                            dist = image_similarity(predictions, outputs, vit, vit_extractor)
                            task_rewards.append(dist / 100)
                        elif 15 <= task_idx <= 104 or 107 <= task_idx <= 184:
                            outputs = list(batch['output'][0])
                            f1 = np.mean(txt_eval(predictions, outputs, bertscore, device=eval_device))
                            task_rewards.append(f1)
                        else:
                            predictions = [pred for pred in predictions]
                            inputs = [text for text in inputs[0]]
                            score = clip_score(predictions, inputs)
                            task_rewards.append(score.detach() / 100)
                    except:
                        task_rewards.append(0.0)
                ave_task_reward = np.mean(task_rewards)
                seq_com.close_module_seq()
                print(f'Score: {ave_task_reward}')
                return_res['reward'] = ave_task_reward
            break

        # check branch
        if len(flow_ptr.branch) == 1:  # no branches
            flow_ptr = list(flow_ptr.branch.values())[0]
        else:
            branch, price = check_branch(client, res, flow_ptr, model_name=args.model_name)
            total_price += price
            flow_ptr = flow_ptr.branch[branch]
        logging.info(f'Current Block: \n```\n{flow_ptr}```')

        plan_round += 1
        
    logging.info(f'The price for task {task_idx} is {total_price}')
    return total_price, return_res

def load_query(args):
    if args.task == 'OpenAGI':
        task_description = ReadLineFromFile("../openagi_data/task_description.txt")
        if 'test' in args.dataset:
            idx_list = [2,3,10,15,20,35,45,55,65,70,90,106,107]
        elif 'train' in args.dataset:
            idx_list = [5, 7, 13, 25, 40, 50, 60, 75, 80, 95, 100, 105, 109]
        else:
            raise NotImplementedError
        return [(i, task_description[i+1]) for i in idx_list]
    else:
        raise NotImplementedError
    
def load_tool(args):
    if args.task == 'OpenAGI':
        return "", {}
    else:
        return "", {}

def load_other_info(args):
    if args.task == 'OpenAGI':
        other_info_list = ReadLineFromFile(args.other_file)
        other_info = '\n'.join(other_info_list)
        return other_info
    else:
        return ""

def main(args, client):
    # args = global_args()
    # args.log_name = os.path.join(args.log_dir, args.log_file_name)
    # set_logger(args)

    # load flow
    # args.flow_file = os.path.join(args.info_dir, args.task, args.flow_name)

    flow = Flow(args.flow_file)
    logging.info(f'```\nFlows:\n{flow}```\n')

    # load task instruction for all query
    instruction_file = os.path.join(args.info_dir, args.task, 'task_instruction.txt')
    if os.path.exists(instruction_file):
        instruction = '\n'.join(ReadLineFromFile(instruction_file))
    else:
        instruction = None
    logging.info(f'```\ntask instruction:\n{instruction}```\n')
    
    # load all query
    task_query = load_query(args)

    # load tool_info and tool_list
    args.tool_file = os.path.join(args.info_dir, args.task, args.tool_name)
    if os.path.exists(args.tool_file):
        tool_info, tool_list = load_tool(args)
    else:
        tool_info, tool_list = "", dict()
    logging.info(f'```\ntool_info:\n{tool_info}```\n')

    # load other_info
    args.other_file = os.path.join(args.info_dir, args.task, args.other_info_name)
    if os.path.exists(args.other_file):
        other_info = load_other_info(args)
    else:
        other_info = ""
    logging.info(f'```\nother_info:\n{tool_info}```\n')

    # Create a notebook to save observations
    notebook = Notebook()

    total_price = 0.0

    if args.task == 'OpenAGI':
        rewards = []
        clips = []
        berts = []
        similairies = []
        valid = []

    if 'gpt' in args.model_name:
        openai_key = args.openai_key
        client = OpenAI(api_key=openai_key)
    elif 'gptq' in args.model_name.lower():
        client = LLM(model=args.model_name, download_dir=args.cache_dir, quantization='gptq', enforce_eager=True, dtype=torch.float16, tensor_parallel_size=8)#, gpu_memory_utilization=0.7)
    else:
        raise NotImplementedError

    return_res = None
    # Answer every query
    for idx, query in task_query:
        try:
            price, return_res = finish_one_task(instruction, tool_info, other_info, flow, idx, query, tool_list, notebook, args, client)
            total_price += price
        except Exception as e:
            logging.error(f"Error when answering {query}: {e}")
        if args.task == 'OpenAGI':
            if return_res is None:
                ave_task_reward = 0.0
            else:
                ave_task_reward = return_res['reward']

            if 0 <= idx <= 14:
                similairies.append(ave_task_reward)
            elif 15 <= idx <= 104 or 107 <= idx <= 184:
                berts.append(ave_task_reward)
            else:
                clips.append(ave_task_reward)

            rewards.append(ave_task_reward)

            if ave_task_reward > 1e-3:
                valid.append(1.0)
            else:
                valid.append(0.0)

    logging.info(f'The price for {args.task} is {total_price}')
    if args.task == 'OpenAGI':
        logging.info(f'Clips: {np.mean(clips)}, BERTS: {np.mean(berts)}, ViT: {np.mean(similairies)}, Rewards: {np.mean(rewards)}, Valid: {np.mean(valid)}')
    return np.mean(rewards)


if __name__ == '__main__':
    args = global_args()
    args.log_name = os.path.join(args.log_dir, args.log_file_name)
    set_logger(args)

    # load flow
    args.flow_file = os.path.join(args.info_dir, args.task, args.flow_name)

    main(args)
