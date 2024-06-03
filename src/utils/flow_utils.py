import os
import logging
import re
import sys
import json
from vllm import LLM, SamplingParams


def ReadLineFromFile(path):
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteDictToFile(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')

def set_logger(args):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_name, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #  Important: Do not print api keys in logs!!!
    logging.info({k: v for k, v in vars(args).items() if 'key' not in k})

def get_observation(client, query, current_progress, notebook, flow_ptr, get_observation_style, model_name):
    if len(notebook.data) == 0:
        return "", "", 0
    if get_observation_style == 'traverse':
        return get_observation_traverse(client, query, current_progress, notebook, flow_ptr, model_name)
    elif get_observation_style == 'direct':
        return get_observation_direct(client, query, current_progress, notebook, flow_ptr, model_name)
    else:
        raise NotImplementedError

def get_observation_traverse(client, query, current_progress, notebook, flow_ptr, model_name):
    progress_str = '\n'.join(current_progress)
    all_observation = notebook.list()
    needed_observation = []
    needed_observation_summary = []
    total_price = 0.0
    for idx, observation in enumerate(all_observation):
        prompt = f'Task description: {query}\n\nCurrent Question: {flow_ptr.get_instruction()}\n\n' \
                 f'Is it helpful to retrieve previous execution results of {observation["Short Description"]} to make informed response of the current question?\n\n' \
                 f'Answer "Yes" or "No" and explain why.'
        if 'gpt' not in model_name:
            prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
        logging.info(f'Observation Check Prompt: \n```\n{prompt}\n```')
        response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, 0., [",", ".", "\\n"])
        logging.info(f'Observation Check Response: \n```\n{response}\n```')
        total_price += price
        if 'yes' in response.lower():
            logging.info(f'Relevant Observation: {observation["key"]} {observation["Short Description"]}')
            needed_observation_summary.append(observation["Short Description"])
            needed_observation.append(notebook.to_str(observation['key']))

    return '\n'.join(needed_observation), '\n'.join(needed_observation_summary), total_price

def get_observation_direct(client, query, current_progress, notebook, flow_ptr, model_name):
    progress_str = '\n'.join(current_progress)
    all_observation = notebook.list()
    needed_observation = []
    needed_observation_summary = []

    prompt = f'Current Progress:\n{progress_str}\n\nTask description: {query}\n\nQuestion: {flow_ptr.get_instruction()}\n\n' \
             f'Select observation that is relevant to answer the current question from the following options:\n' 
    for i, observation in enumerate(all_observation):
        prompt += f'{i + 1}: {observation["key"]} {observation["Short Description"]}.\n'

    prompt += 'Your answer should be numbers separated by commas, referring to the desired choice, with no additional text or explanations. Do not be verbose.'
    # logging.info(f'Prompt: \n```\n{prompt}\n```')
    if 'gpt' not in model_name:
        prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
    response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, 0., ['.'])
    logging.info(f'Relevant Observation: \n```\n{response}\n```')
    idx_list = [idx.strip() for idx in response.strip().split(',')]
    for idx in idx_list:
        if idx.isdigit() and 1 <= int(idx) <= len(all_observation):
            needed_observation_summary.append(all_observation[int(idx)-1]['Short Description'])
            needed_observation.append(notebook.to_str(all_observation[int(idx)-1]['key']))
    return '\n'.join(needed_observation), '\n'.join(needed_observation_summary), price


def get_prompt(tool_info, flow_ptr, task_description, cur_progress, observations="", model_name='gpt-4-1106-preview', other_info=""):
    progress_str = '\n'.join(cur_progress)
    if 'gpt' not in model_name:
        if len(observations) > 0:
            prompt = f'[INST] ### {tool_info}\n\n{other_info}\n\nCurrent Progress:\n{progress_str}\n\nTask description: {task_description}\n\nObservations:\n{observations}\n\n' \
                f'Question: {flow_ptr.get_instruction()}\n\nOnly answer the current instruction and do not be verbose. \n\n### Answer:\n [\INST]'
        else:
            prompt = f'[INST] ### {tool_info}\n\n{other_info}\n\nCurrent Progress:\n{progress_str}\n\nTask description: {task_description}\n\n' \
                f'Question: {flow_ptr.get_instruction()}\n\nOnly answer the current instruction and do not be verbose. \n\n### Answer:\n [\INST]'
    else:
        if len(observations) > 0:
            prompt = f'{tool_info}\n\n{other_info}\n\nCurrent Progress:\n{progress_str}\n\nTask description: {task_description}\n\nObservations:\n{observations}\n\n' \
                f'Question: {flow_ptr.get_instruction()}\n\nOnly answer the current instruction and do not be verbose.'
        else:
            prompt = f'{tool_info}\n\n{other_info}\n\nCurrent Progress:\n{progress_str}\n\nTask description: {task_description}\n\n' \
                f'Question: {flow_ptr.get_instruction()}\n\nOnly answer the current instruction and do not be verbose.'
    return prompt


def check_branch(client, messages, flow_ptr, model_name, temperature=0.):
    possible_keys = list(flow_ptr.branch.keys())
    prompt = f'Given the question ```{flow_ptr.get_instruction()}```, choose the closest representation of ```{messages}``` from the following options:\n'
    for i, key in enumerate(possible_keys):
        prompt += f'{i + 1}: {key}.\n'
    prompt += "Your answer should be only an number, referring to the desired choice. Don't be verbose!"
    if 'gpt' not in model_name:
        prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
    # logging.info(f'Prompt: \n```\n{prompt}\n```')
    total_price = 0.0
    while True:
        response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, temperature, ['.'])
        total_price += price
        temperature += .5
        if response.isdigit() and 1 <= int(response) <= len(possible_keys):
            response = int(response)
            break
        print(f'Temperature: {temperature}')
        if temperature > 2:
            print('No valid format output when calling "Check Branch".')
            exit(1)
    logging.info(f'{response}, {possible_keys[response - 1]}')
    return possible_keys[response - 1], total_price

def notebook_summarize(client, tool_info, tool_calling, model_name, temperature=0.):
    prompt = f"{tool_info}\n\n" \
             f"For the information of ```{tool_calling}```, describe the provided information in a concise format without mentioning the tool's name." \
             f"The arguments should be exacttly included in the description. "
    if 'gpt' not in model_name:
        prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
    response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, temperature, ['.'])
    logging.info(f'Short Summary for {tool_calling}: {response}')
    return response, price
    

def check_tool_use(client, observation, flow_ptr, messages, tool_info, model_name, temperature=0.):
    prompt = f'You are allowed to use the following tools: \n\n```{tool_info}```\n\n' \
             f'Initial Response: ```{messages}```\n' \
             f'Current Question: ```{flow_ptr.get_instruction()}```\n\n' \
             f'Is there a need to execute additional provided tools to fully address the current question? \n\n' \
             f'Answer "Yes" or "No" and explain why. '
    logging.info(f'Tool Use Prompt: \n```\n{prompt}\n```')
    # f'Existing Tool Usage: ```{observation}```\n\n' \
    #          f'You may execute tools with the same name if the arguments are different, but do not execute any tool with the same name and the same arguments that have been used previously.\n' \
    if 'gpt' not in model_name:
        prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
    total_price = 0.0
    while True:
        response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, temperature, [",", ".", "\\n"])
        total_price += price
        temperature += .5
        logging.info(f'Tool use check: {response}')
        if 'yes' in response.lower():
            return True, total_price
        if 'no' in response.lower():
            return False, total_price
        print(f'Temperature: {temperature}')
        if temperature > 2:
            break
    logging.info('No valid format output when calling "Tool use check".')
    exit(1)


def get_tool_arg(client, flow_ptr, messages, tool_info, selected_tool, model_name):
    prompt = f'{tool_info}\n\n' \
             f'You attempt to answer the question ```{flow_ptr.get_instruction()}```. ' \
             f'Initial Response: ```{messages}```? ' \
             f'What is the input argument for the tool ```{selected_tool}``` given the initial response? ' \
             f'Respond "None" if no arguments are needed for this tool. Separate by comma if there are multiple arguments. Do not be verbose!'
    # logging.info(f'Prompt: \n```\n{prompt}\n```')
    # f'What is the input argument to call tool for this step: ```{messages}```? ' \
    if 'gpt' not in model_name:
        prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
    response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, 0.)
    logging.info(f'Parameters: {response}')
    return response, price

def check_tool_name(client, flow_ptr, messages, tool_list, model_name, temperature=0.):
    prompt = f'Given the question ```{flow_ptr.get_instruction()}```, choose the used tool of ```{messages}``` from the following options:\n'
    for i, key in enumerate(tool_list):
        prompt += f'{i + 1}: {key}.\n'
    prompt += "Your answer should be only an number, referring to the desired choice. Don't be verbose!"
    # logging.info(f'Prompt: \n```\n{prompt}\n```')
    if 'gpt' not in model_name:
        prompt = f'### Question:\n{prompt}\n\n### Answer:\n'
    total_price = 0.0
    while True:
        response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, temperature, ['.'])
        total_price += price
        temperature += .5
        if response.isdigit() and 1 <= int(response) <= len(tool_list):
            response = int(response)
            break
        print(f'Temperature: {temperature}')
        if temperature > 2:
            logging.info('No valid format output when calling "Tool name select".')
            exit(1)
    logging.info(f'{response}, {tool_list[response - 1]}')
    return tool_list[response - 1], total_price

def get_response_from_client(client, messages, model_name, temperature=1., stop=None):
    if 'gpt' not in model_name:
        return get_response_from_vllm(client, messages, model_name, temperature, stop)
    else:
        return get_response_from_gpt(client, messages, model_name, temperature, stop)


def get_response_from_vllm(client, messages, model_name, temperature=1., stop=None):
    if stop is None:
        stop = []
    sampling_param = SamplingParams(temperature=temperature, max_tokens=1024, stop=['###']+stop)
    if messages[0]['role'] == 'user':
        res = client.generate(messages[0]['content'], sampling_params=sampling_param)[0].outputs[0].text.strip().replace('```', '"')
    else:
        res = client.generate(f"{messages[0]['content']}\n{messages[1]['content']}", sampling_params=sampling_param)[0].outputs[0].text.strip().replace('```', '"')
    
    return res, 0.0


def get_response_from_gpt(client, messages, model_name, temperature=1., stop=None):
    if stop is None:
        stop = []
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop,
    )
    total_prompt_tokens = response.usage.prompt_tokens
    total_completion_tokens = response.usage.completion_tokens
    price = calc_cost_w_prompt(total_prompt_tokens, model_name) + calc_cost_w_completion(total_completion_tokens, model_name)
    return response.choices[0].message.content, price

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s


def openai_unit_price(model_name,token_type="prompt"):
    if 'gpt-4-' in model_name:
        if token_type=="prompt":
            unit = 0.01
        elif token_type=="completion":
            unit = 0.03
        else:
            raise ValueError("Unknown type")
    elif 'gpt-4' in model_name:
        if token_type=="prompt":
            unit = 0.03
        elif token_type=="completion":
            unit = 0.06
        else:
            raise ValueError("Unknown type")
    elif 'gpt-3.5-turbo' in model_name:
        unit = 0.002
    elif 'davinci' in model_name:
        unit = 0.02
    elif 'curie' in model_name:
        unit = 0.002
    elif 'babbage' in model_name:
        unit = 0.0005
    elif 'ada' in model_name:
        unit = 0.0004
    else:
        unit = -1
    return unit


def calc_cost_w_completion(total_tokens: int, model_name: str):
    unit = openai_unit_price(model_name,token_type="completion")
    return round(unit * total_tokens / 1000, 4)


def calc_cost_w_prompt(total_tokens: int, model_name: str):
    # 750 words == 1000 tokens
    unit = openai_unit_price(model_name)
    return round(unit * total_tokens / 1000, 4)
