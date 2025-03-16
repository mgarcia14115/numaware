import torch
from transformers import BertTokenizer
from uafscs.configs import defaults as config


print("Please enter a review. It will be classified as positive, negative or neutral.")

review = input()


device = config.DEVICE

def label_converter(idx):
	if idx == 0:
		return "negative"
	elif idx == 2:
		return "positive"
	else:
		return "neutral"

model = torch.load("../../local/models/2025-01-03_09-40-19.pth",weights_only=False,map_location=device)

model = model.to(device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


encoded_input = tokenizer(review, return_tensors="pt",max_length=config.DEFAULTS["max_seq_length"],padding=False,truncation=True)

input = encoded_input['input_ids'].squeeze(0).to(device)

output = model(input)

_,idx = torch.max(output,0)

print(label_converter(idx))