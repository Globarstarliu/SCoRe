import json
import os

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

ann_path = 'assets/okvqa/test/annotations/mscoco_val2014_annotations.json'
que_path = 'assets/okvqa/test/questions/OpenEnded_mscoco_val2014_questions.json'

relation_path = 'assets/okvqa/test/relation/triple_rel_ok.json'
with open(relation_path, 'r', encoding='utf-8') as f:
	relations = json.load(f)
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
is_cuda_available = torch.cuda.is_available()
cuda_devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
print("Is CUDA available?", is_cuda_available)
print(f"CUDA devices: {cuda_devices}")

model_file = 'mistralai/Mistral-7B-v0.1'
model = AutoModelForCausalLM.from_pretrained(model_file, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_file)

model.eval()

sum = 0
acc_avg = 0.0

print("Model loaded, loading data...")
dataset = OKVQADataset(que_path, ann_path)

for idx, sample in enumerate(dataset):
	head = "Strictly answer with ONE exact word from the context (excluding proper nouns). No explanations or compound terms.\n"
	question = sample["question"]
	question_id = sample['question_id']

	relation = relations[str(question_id)]["relation"]
	relation = [f"{tuple(r)[0]} {tuple(r)[2]} {tuple(r)[1]}".replace("'", "").replace('"', "") for r in relation if len(r) == 3]
	rel_str = "; ".join(relation)

	prompt = (head + "relationships: " + rel_str + "\n" +
	          "question:" + question + "\n" + "answer:")
	if not idx:
		print(prompt)

	encoded = tokenizer(prompt, return_tensors="pt")
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
	sum = sum + acc
	acc_avg = sum / (idx + 1)

	print(f"{idx + 1}        predict:{pred_answer}, ground truth:{sample['answers'][0]['answer']}")
	print(f"                acc:{acc}, acc_avg:{acc_avg}")

	result_file = 'result/okvqa/zero_shot_score.txt'
	with open(result_file, 'a', encoding='utf-8') as file:
		file.write(
			f"{idx + 1} predict:{pred_answer}, sum:{sum}, answer1:{sample['answers'][0]['answer']}, acc:{acc}, acc_avg:{acc_avg}\n")

