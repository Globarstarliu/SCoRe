import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
from my_dataset.okvqa_datasets import OKVQADataset
import inflect
import numpy as np
import clip
from PIL import Image
from tqdm import tqdm
ann_path = 'assets/okvqa/test/annotations/mscoco_val2014_annotations.json'
que_path = 'assets/okvqa/test/questions/OpenEnded_mscoco_val2014_questions.json'
img_path = 'assets/okvqa/test/val2014'

ann_path_train = 'assets/okvqa/train/annotations/mscoco_train2014_annotations.json'
que_path_train = 'assets/okvqa/train/questions/OpenEnded_mscoco_train2014_questions.json'
img_path_train = 'assets/okvqa/train/train2014'
dataset_train = OKVQADataset(que_path_train, ann_path_train)
dataset = OKVQADataset(que_path, ann_path)

def encoder_text(text, max_len=77):
    text = text[:max_len * 4]
    text_inputs = clip.tokenize(text, truncate=True).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs).float()
    return text_feature

def encoder_img(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input).float()
    return image_feature
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/16', device, jit=False)

features_train = {}
for idx, sample in tqdm(enumerate(dataset_train), total=len(dataset_train)):
    question = sample['question']
    question_id = sample['question_id']
    image_id = sample['image_id']
    image_path = img_path_train + "/COCO_train2014_" + str(image_id).zfill(12) + ".jpg"
    image = Image.open(image_path)
    que_feature = encoder_text(question).cpu().numpy()
    img_feature = encoder_img(image).cpu().numpy()
    features_train[str(question_id)] = que_feature
    features_train[f"img_{question_id}"] = img_feature

np.savez('features_okvqa_train.npz', **features_train)

features_test = {}
for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
    question = sample['question']
    question_id = sample['question_id']
    image_id = sample['image_id']
    image_path = img_path + "/COCO_val2014_" + str(image_id).zfill(12) + ".jpg"
    image = Image.open(image_path)
    que_feature = encoder_text(question).cpu().numpy()
    img_feature = encoder_img(image).cpu().numpy()
    features_test[str(question_id)] = que_feature
    features_test[f"img_{question_id}"] = img_feature

np.savez('features_okvqa_test.npz', **features_test)

del model, preprocess
torch.cuda.empty_cache()

relation_path = 'assets/okvqa/test/relation/triple_rel_ok.json'
with open(relation_path, 'r', encoding='utf-8') as f:
    relations = json.load(f)
relation_path_train = 'assets/okvqa/train/relation/triple_rel_ok.json'
with open(relation_path_train, 'r', encoding='utf-8') as f:
    relations_train = json.load(f)
def extract_string(text):
    dot_index = text.find(".")
    newline_index = text.find("\n")
    parenthesis_index = text.find(" (")
    comma_index = text.find(",")
    index_or = text.find(" or ")

    indices = [index for index in [dot_index, newline_index, parenthesis_index, comma_index, index_or] if index != -1]
    if indices:
        index = min(indices)
    else:
        index = -1

    if index != -1:
        return text[:index]
    else:
        return text

def remove_prefix(word):
    if word.startswith("a "):
        return word[2:]
    elif word.startswith("an "):
        return word[3:]
    elif word.startswith("the "):
        return word[4:]
    else:
        return word

def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    tagged_word = nltk.pos_tag([word])
    pos = tagged_word[0][1][0].lower()

    if pos in ['v', 'n']:
        if word.endswith("ing"):
            return lemmatizer.lemmatize(word, pos='v')
        return lemmatizer.lemmatize(word, pos=pos)
    else:
        return lemmatizer.lemmatize(word)

def norm_ans(ans):
    ans = extract_string(ans)
    ans = remove_prefix(ans)
    single = inflect.engine()
    if ans.endswith(',') or ans.endswith('.'):
        ans = ans[:-1]
    if ans.startswith('a '):
        ans = ans[2:]
    if ',' in ans:
        ans = ans.split(',')[0]
    if ans and len(ans.split(' ')) == 1 and ans not in ['bus', 'yes'] and not ans.endswith("ss"):
        norm_ans = single.singular_noun(ans)
        if norm_ans != False:
            ans = norm_ans
    ans = lemmatize_word(ans)
    return ans.replace("-", " ").lower()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
is_cuda_available = torch.cuda.is_available()
cuda_devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
print("Is CUDA available?", is_cuda_available)
print(f"CUDA devices: {cuda_devices}")
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model_file = 'mistralai/Mistral-7B-v0.1'
model = AutoModelForCausalLM.from_pretrained(model_file, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_file)

model.eval()
sum_4 = 0
acc_avg_4 = 0.0
sum_8 = 0
acc_avg_8 = 0.0
sum_16 = 0
acc_avg_16 = 0.0

print("Model loaded, loading data...")

test_feature = np.load('features_okvqa_test.npz')
train_feature = np.load('features_okvqa_train.npz')
print("Dataset length:", len(dataset))
for idx, sample in enumerate(dataset):
    head = "Strictly answer with ONE exact word from the context (excluding proper nouns). No explanations or compound terms.\n"
    question = sample["question"]
    question_id = sample['question_id']

    relation = relations[str(question_id)]["relation"]
    relation = [f"{tuple(r)[0]} {tuple(r)[2]} {tuple(r)[1]}".replace("'", "").replace('"', "") for r in relation if len(r) == 3]
    rel_str = "; ".join(relation)

    similarity_list = []
    que_feature = torch.from_numpy(test_feature[str(question_id)])
    img_feature = torch.from_numpy(test_feature[f"img_{question_id}"])
    criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    for idx_train, sample_train in enumerate(dataset_train):
        que_feature_train = torch.from_numpy(train_feature[str(sample_train['question_id'])])
        img_feature_train = torch.from_numpy(train_feature[f"img_{sample_train['question_id']}"])
        similarity_que = criterion(que_feature, que_feature_train)
        similarity_img = criterion(img_feature, img_feature_train)
        similarity = (similarity_que.item() + similarity_img.item()) / 2
        similarity_list.append((similarity, idx_train))
    similarity_list.sort(reverse=True, key=lambda x: x[0])
    top32_similar = similarity_list[:32]
    few_shot_examples_32 = [dataset_train[idx] for (sim, idx) in top32_similar]
    few_shot_4 = ""
    few_shot_8 = ""
    few_shot_16 = ""
    for idx_sample, sample_32 in enumerate(few_shot_examples_32):
        q = sample_32['question']
        a = sample_32['answers'][0]['answer']
        train_question_id = sample_32['question_id']
        relation_idx = relations_train[str(train_question_id)]["relation"]
        relation_idx = [f"{tuple(r)[0]} {tuple(r)[2]} {tuple(r)[1]}".replace("'", "").replace('"', "") for r in relation_idx if
                    len(r) == 3]
        relation_idx = "; ".join(relation_idx)
        if idx_sample < 4:
            few_shot_4 = few_shot_4 + "relationships: " + relation_idx + "\nquestion: " + q + "\nanswer: " + a + "\n"
        if idx_sample < 8:
            few_shot_8 = few_shot_8 + "relationships: " + relation_idx + "\nquestion: " + q + "\nanswer: " + a + "\n"
        if idx_sample < 16:
            few_shot_16 = few_shot_16 + "relationships: " + relation_idx + "\nquestion: " + q + "\nanswer: " + a + "\n"
    prompt_4 = (head + few_shot_4 + "relationships: " + rel_str + "\n" + "question: " + question + "\n" + "answer:")
    prompt_8 = (head + few_shot_8 + "relationships: " + rel_str + "\n" + "question: " + question + "\n" + "answer:")
    prompt_16 = (head + few_shot_16 + "relationships: " + rel_str + "\n" + "question: " + question + "\n" + "answer:")
    if not idx:
        print(prompt_4)
    if not idx:
        print(prompt_8)
    if not idx:
        print(prompt_16)
    encoded = tokenizer(prompt_4, return_tensors="pt")
    model_inputs = encoded.to(device)
    generate_ids = model.generate(
        input_ids=model_inputs.input_ids,
        max_length=len(model_inputs.input_ids[0]) + 50,
        pad_token_id=tokenizer.eos_token_id,
    )
    pred_answer = tokenizer.batch_decode(generate_ids[:, len(model_inputs.input_ids[0]):])[0]
    if pred_answer:
        pred_answer = norm_ans(pred_answer)
    else:
        pred_answer = "unknown"
    true_times = 0
    for answer in sample["answers"]:
        if pred_answer == answer['answer']:
            true_times += 1

    acc = min(1.0, true_times / 3.0)
    sum_4 = sum_4 + acc
    acc_avg_4 = sum_4 / (idx + 1)

    result_file = 'result/okvqa/few_shot_4_score.txt'
    with open(result_file, 'a', encoding='utf-8') as file:
        file.write(
            f"{idx + 1} predict:{pred_answer}, sum:{sum_4}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg_4}\n")
    print(f"shot4-{idx + 1} predict:{pred_answer}, sum:{sum_4}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg_4}\n")    
    
    encoded = tokenizer(prompt_8, return_tensors="pt")
    model_inputs = encoded.to(device)
    generate_ids = model.generate(
        input_ids=model_inputs.input_ids,
        max_length=len(model_inputs.input_ids[0]) + 50,
        pad_token_id=tokenizer.eos_token_id,
    )
    pred_answer = tokenizer.batch_decode(generate_ids[:, len(model_inputs.input_ids[0]):])[0]
    if pred_answer:
        pred_answer = norm_ans(pred_answer)
    else:
        pred_answer = "unknown"
    true_times = 0
    for answer in sample["answers"]:
        if pred_answer == answer['answer']:
            true_times += 1

    acc = min(1.0, true_times / 3.0)
    sum_8 = sum_8 + acc
    acc_avg_8 = sum_8 / (idx + 1)

    result_file = 'result/okvqa/few_shot_8_score.txt'
    with open(result_file, 'a', encoding='utf-8') as file:
        file.write(
            f"{idx + 1} predict:{pred_answer}, sum:{sum_8}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg_8}\n")
    print(f"shot8-{idx + 1} predict:{pred_answer}, sum:{sum_8}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg_8}\n")

    encoded = tokenizer(prompt_16, return_tensors="pt")
    model_inputs = encoded.to(device)
    free, total = torch.cuda.mem_get_info()
    print(f"free={free/1024**3:.2f} GiB, total={total/1024**3:.2f} GiB")
    generate_ids = model.generate(
        input_ids=model_inputs.input_ids,
        max_length=len(model_inputs.input_ids[0]) + 50,
        pad_token_id=tokenizer.eos_token_id,
    )
    pred_answer = tokenizer.batch_decode(generate_ids[:, len(model_inputs.input_ids[0]):])[0]
    if pred_answer:
        pred_answer = norm_ans(pred_answer)
    else:
        pred_answer = "unknown"
    true_times = 0
    for answer in sample["answers"]:
        if pred_answer == answer['answer']:
            true_times += 1

    acc = min(1.0, true_times / 3.0)
    sum_16 = sum_16 + acc
    acc_avg_16 = sum_16 / (idx + 1)

    result_file = 'result/okvqa/few_shot_16_score.txt'
    with open(result_file, 'a', encoding='utf-8') as file:
        file.write(
            f"{idx + 1} predict:{pred_answer}, sum:{sum_16}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg_16}\n")
    print(f"shot16-{idx + 1} predict:{pred_answer}, sum:{sum_16}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg_16}\n")
    