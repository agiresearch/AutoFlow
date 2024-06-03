from evaluate import load
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

def txt_eval(predictions, references, bertscore, device="cuda"):
    score = bertscore.compute(
                    predictions=predictions,
                    references=references,
                    lang="en",
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=device)["f1"]
    
    return score


def txt_loader(path):
    text = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            text.append(line)
    f.close()
    return text


def image_similarity(im1, im2, model, extractor):
    batch_size = len(im1)
    # Load two images
    img1 = extractor(im1, return_tensors="pt")
    img2 = extractor(im2, return_tensors="pt")

    # Preprocess the images and get their embeddings
    with torch.no_grad():
        emb1 = model(img1.pixel_values)[0].squeeze().numpy()
        emb2 = model(img2.pixel_values)[0].squeeze().numpy()

    # Compute the cosine similarity between the embeddings
    dist = np.mean(np.array([np.linalg.norm(emb1[i] - emb2[i], ord='fro') for i in range(batch_size)]))
    return dist

def module_seq_filter(module_seq, task_id):
    io_dict = { 
                "Colorization":['image','image'],  
                "Image Denoising":['image','image'], 
                "Image Deblurring":['image','image'],
                "Image Super Resolution":['image','image'],
                "Image Classification":['image','text'],  
                "Image Captioning":['image','text'], 
                "Object Detection":['image','text'],
                "Text Summarization":['text','text'],  
                "Text Generation":['text','text'], 
                "Machine Translation":['text','text'],  
                "Fill Mask":['text','text'],
                "Sentiment Analysis":['text','text'],
                "Text to Image Generation":['text','image'],
                "Question Answering":['text-text','text'],
                "Visual Question Answering":['image-text','text']
        }
    module_seq_list = module_seq.split(", ")
    input_type = io_dict[module_seq_list[0]][0]
    output_type = io_dict[module_seq_list[-1]][1]
    if input_type == "image" and output_type == "image" and 0<=task_id<=14:
        return True
    elif input_type == "image" and output_type == "text" and 15<=task_id<=104:
        return True
    elif input_type == "text" and output_type == "image" and 105<=task_id<=107:
        return True
    elif input_type == "text" and output_type == "text" and 108<=task_id<=125:
        return True
    elif input_type == "image-text" and output_type == "text" and 126<=task_id<=170:
        return True
    elif input_type == "text-text" and output_type == "text" and 171<=task_id<=188:
        return True
    else:
        return False
    
    
def whole_module_seq_filter(module_seq, task_id):
    io_dict = { 
                "Colorization":['image','image'],  
                "Image Denoising":['image','image'], 
                "Image Deblurring":['image','image'],
                "Image Super Resolution":['image','image'],
                "Image Classification":['image','text'],  
                "Image Captioning":['image','text'], 
                "Object Detection":['image','text'],
                "Text Summarization":['text','text'],  
                "Text Generation":['text','text'], 
                "Machine Translation":['text','text'],  
                "Fill Mask":['text','text'],
                "Sentiment Analysis":['text','text'],
                "Text to Image Generation":['text','image'],
                "Question Answering":['text-text','text'],
                "Visual Question Answering":['image-text','text']
        }
    module_seq_list = module_seq.split(", ")
    condition_1 = None
    for i, m in enumerate(module_seq_list):
        if i < len(module_seq_list)-1 and io_dict[m][1] != io_dict[module_seq_list[i+1]][0]:
            condition_1 = False
            break
        else:
            condition_1 = True
            
        
    condition_2 = None   
    input_type = io_dict[module_seq_list[0]][0]
    output_type = io_dict[module_seq_list[-1]][1]
    if input_type == "image" and output_type == "image" and 0<=task_id<=14:
        condition_2 = True
    elif input_type == "image" and output_type == "text" and 15<=task_id<=104:
        condition_2 = True
    elif input_type == "text" and output_type == "image" and 105<=task_id<=107:
        condition_2 = True
    elif input_type == "text" and output_type == "text" and 108<=task_id<=125:
        condition_2 = True
    elif input_type == "image-text" and output_type == "text" and 126<=task_id<=170:
        condition_2 = True
    elif input_type == "text-text" and output_type == "text" and 171<=task_id<=188:
        condition_2 = True
    else:
        condition_2 = False
        
    return condition_1 and condition_2
    
    
    
def match_module_seq(model_steps, sentence_model):
    module_seq = ""

    for i in range(len(model_steps)):

        sentences1 = [model_steps[i]]*15

        sentences2 = ["Image Classification","Colorization","Object Detection",\
                  "Image Super Resolution","Image Captioning","Image Deblurring",\
                  "Image Denoising","Text to Image Generation","Visual Question Answering",\
                  "Sentiment Analysis","Question Answering","Text Summarization",\
                  "Text Generation","Machine Translation","Fill Mask"]

        #Compute embedding for both lists
        embeddings1 = sentence_model.encode(sentences1, convert_to_tensor=True)#.to(device_)
        embeddings2 = sentence_model.encode(sentences2, convert_to_tensor=True)#.to(device_)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        similarities = torch.stack([cosine_scores[i][i] for i in range(15)])

        module_index = torch.argmax(similarities).item()
        module_seq += sentences2[module_index] + ","
        # print(similarities[module_index])
        # print(sentences2[module_index])

    #Output the pairs with their score
    # for i in range(len(sentences1)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    module_seq = module_seq.strip()[:-1]
    return module_seq


def parse_module_list_with_gpt(args, generated_module_seq):
    openai_key = args.openai_key
    client = OpenAI(api_key=openai_key)
    todo_prompt = f"You are a key phrase extractor who is able to extract potential module names from the given " \
                  f"context. You have already known all the module names in the full module list. The full module " \
                  f"list is: [Image Classification, Colorization, Object Detection, Image Deblurring, " \
                  f"Image Denoising, Image Super Resolution, Image Captioning, Text-to-Image Generation, " \
                  f"Visual Question Answering, Sentiment Analysis, Question Answering, Text Summarization, " \
                  f"Machine Translation]. Given the following context: ```{generated_module_seq}```. Please extract " \
                  f"a module sequence from this context and remove module names which do not exist in the full " \
                  f"module list from this sequence. Output the module sequence after filtering as the format of " \
                  f"'module: module1, module: module2, module: module3, etc...'. "

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": todo_prompt}
        ]
    )

    response = response.choices[0].message.content

    # response = llm.generate(todo_prompt)[0].outputs[0].text

    print(f'Parsed response:\n```{response}\n```')

    response = response.split("module: ")[1:]

    result = ""
    for c in response:
        result += c

    print(f'Module List:\n```{result}\n```')
    # result = result[:-1] if len(result) > 0 else result

    return result