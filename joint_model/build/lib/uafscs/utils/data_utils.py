import torch
from uafscs.configs import defaults as config
class myDataset(torch.utils.data.Dataset):
	
	def __init__(self , df,tokenizer):
		
		self.tokenizer = tokenizer
		self.df = df

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		text = self.tokenizer(self.df["text"][idx],return_tensors="pt",max_length=config.DEFAULTS["max_seq_length"],padding=False,truncation=True)["input_ids"].squeeze(0)
		label = torch.tensor(int(self.df["label"][idx]),dtype=torch.long).unsqueeze(0)
		return text,label

class CustomDataset(torch.utils.data.Dataset):

	def __init__(self , dataframe = None , **kwargs):		
		self.x = dataframe["text"]
		self.y = dataframe["label"]
	
	def __len__(self):
		return len(self.x)
	
	def __getitem__(self, index = None , **kwargs):
		
		text  = self.x[index]
		label = self.y[index]
		return{
				"text" :text,
				"label":label
		      }
	


def convert_labels(label):
	label = int(label)
	if label == 0 or label == 1:
		return 0
	elif label == 3 or label == 4:
		return 2
	else:
		return 1