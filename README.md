# AutoFlow: Automated Workflow Generation for Large Language Model Agents

Recent advancements in Large Language Models (LLMs) have shown significant progress in understanding complex natural language. However, LLMs still face challenges in generating and executing programming codes accurately. While some efforts have been made to leverage LLMs for code generation, many generated codes are still unable to be executed effectively. In contrast, natural language programs can have a higher executable rate due to their minimal syntax requirements compared to traditional programming languages. As evidence, recent work proposes the CoRE language, utilizing LLMs as interpreters for workflow programming via natural language, achieving a higher valid plan rate compared to baseline methods. However, CoRE requires manual design and may lead to suboptimal solutions. To address these issues, we propose AutoFlow, a framework designed to automatically generate workflows in the CoRE language for solving complex tasks. AutoFlow offers two workflow generation methods: fine-tuning-based and in-context-based methods, making it applicable to both open-source and closed-source LLMs. Our framework produces more robust and reliable workflows than existing code generation methods. Moreover, natural language programming offers greater readability and lower barriers for coders than traditional programming languages. We believe that the automatic generation and interpretation of workflows in natural language represent a promising paradigm for solving complex tasks, particularly with the rapid development of LLMs.

This package is mainly contributed by [Zelong Li](https://github.com/lzl65825) (zelong.li@rutgers.edu), [Shuyuan Xu](https://github.com/shuyuan-x) (shuyuan.xu@rutgers.edu), and [Yongfeng Zhang](https://github.com/evison) (yongfeng.zhang@rutgers.edu). We welcome any issues and requests for model implementation and bug fix.

## Citation

To be updated

## Requirements

- Python==3.9
- PyTorch==2.2.2
- transformers==4.40.2
- langchain==0.1.4
- peft==0.7.1

## Preparation

0. Clone this repo.

1. Create a conda virtual environment and install the Pytorch matching your CUDA version. For example, for CUDA version 12.1:

```
conda create -n your_env_name python=3.9
conda activate your_env_name

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

2. Install necessary packages:

```
pip install -r requirements.txt
```

3. Download the OpenAGI data from this [Google Drive link](https://drive.google.com/drive/folders/1AjT6y7qLIMxcmHhUBG5IE1_5SnCPR57e?usp=share_link), unzip it to the `AutoFlow` directory and rename it as `openagi_data`.

4. Make sure you are in the *AutoFlow/src* folder before running the codes. Otherwise,

```
cd src
```

## Running Command Examples

OpenAGI on gpt-4-1106-preview:
```commandline
python auto_agi.py 
--model_name="gpt-4-1106-preview"
--auto_model_name="gpt-4-1106-preview"
--log_file_name=../log/autoagi_gpt4gpt.txt
--output_dir=./gpt4gpt
--auto_flow_name="autoagi_gpt4gpt_Flow.txt"
--openai_key="YOUR OPENAI KEY"
```

OpenAGI on TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ:
```commandline
python auto_agi.py 
--model_name="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
--auto_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
--log_file_name=../log/autoagi_mixtral4mixtral.txt
--output_dir=./mixtral4mixtral
--auto_flow_name="autoagi_mixtral4mixtral_Flow.txt"
--openai_key="YOUR OPENAI KEY"
```

## Reference

- We leveraged the dataset of [OpenAGI](https://github.com/agiresearch/OpenAGI) projects and based on [CoRE language] (https://github.com/agiresearch/CoRE) to implement our experiment.
