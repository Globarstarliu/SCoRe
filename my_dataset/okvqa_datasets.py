import json
from torch.utils.data import Dataset


class OKVQADataset(Dataset):
	def __init__(self, que_path, ann_path, cap_path=None):
		self.annotations = self.load_annotations(ann_path)
		self.questions = self.load_questions(que_path)

	def load_annotations(self, ann_path):
		with open(ann_path, 'r', encoding='utf-8') as f:
			anns = json.load(f)['annotations']
		return anns
	def load_questions(self, que_path):
		with open(que_path, 'r', encoding='utf-8') as f:
			ques = json.load(f)['questions']
		return ques

	def __len__(self):
		return len(self.questions)

	def __getitem__(self, idx):
		question = self.questions[idx]
		image_id = question['image_id']
		question_text = question['question']
		answers = [ann['answers'] for ann in self.annotations if ann['question_id'] == question['question_id']][0]
		return {
			'question': question_text,
			'answers': answers,
			'image_id': f"000000{image_id:06d}",
			'question_id': question['question_id'],
		}



