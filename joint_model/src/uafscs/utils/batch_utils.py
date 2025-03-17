# from transformers import BertTokenizer
# import torch

# from uafscs.configs import defaults as config


# def rnn_collate_fn(batch):
	
# 	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# 	inputs = []
# 	labels = []

# 	for record in batch:
# 		inputs.append(record["text"])
# 		labels.append(int(record["label"]))


# 	encoder = tokenizer(inputs, return_tensors="pt" , max_length = config.DEFAULTS["max_seq_length"] , padding = "max_length" , truncation = True)
# 	labels  = torch.tensor(labels, dtype=torch.long)
	
# 	return {
# 			 "inputs"        :encoder["input_ids"],
# 			 "labels"        :labels,
# 			 "attention_mask":encoder["attention_mask"]
# 		   }
	
	