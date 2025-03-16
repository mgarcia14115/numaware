from tqdm import tqdm
import torch
import uafscs.utils.metrics_utils	    as metils
from sklearn.metrics import classification_report
from uafscs.configs import defaults as config
import random
import numpy as np

class UAFSTrainer():

	def __init__( self , 
			      model        = None , 
				  optimizer    = None ,
				  loss_fn      = None ,
				  train_dataset= None ,
				  test_dataset = None , 
				  train_loader = None ,
				  test_loader  = None ,
				  epochs       = None ,
				 **kwargs
				):
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.model        = model.to(config.DEVICE)
		self.optimizer    = optimizer
		self.loss_fn      = loss_fn
		self.train_loader = train_loader
		self.test_loader  = test_loader
		self.epochs       = epochs

	def train(self):
		self.model.train()
		loss_per_epoch = []
		y_true = []
		y_pred = []
		target_names = ["negative" , "neutral" , "positive"]
		for epoch in range(self.epochs):
			epoch_loss = 0
			for batch in tqdm(self.train_loader):
				inputs = batch["inputs"]
				labels = batch["labels"]
				
				inputs  = inputs.to(config.DEVICE)
				labels  = labels.to(config.DEVICE)

				outputs = self.model(inputs)
				_,idxs  = torch.max(outputs,1)

				y_true.extend(labels.tolist())
				y_pred.extend(idxs.tolist())

				loss = self.loss_fn(outputs,labels)
				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.item()
			
			avg_epoch_loss = epoch_loss / len(self.train_loader)

			loss_per_epoch.append(epoch_loss)
			
			print(f"Epoch: {epoch + 1}    Loss: {round(avg_epoch_loss,3)}")
			print(classification_report(y_true=y_true,y_pred=y_pred,target_names= target_names, zero_division= 0))
			
	

	
		
	def train_updated(self,batch_size):
		self.model.train()
		loss_per_epoch = []
		y_true = []
		y_pred = []
		
		target_names = ["negative" , "neutral" , "positive"]
		for epoch in range(self.epochs):
			epoch_loss = 0


			batches = list(range(len(self.train_dataset))) # -> train_dataset = 10 ,,, [0,1,2,3,4,5,6,7,8,9] = batches
			random.shuffle(batches)
			batches = np.array_split(batches, len(batches) // batch_size)

			for idx,batch in tqdm(enumerate(batches), total=len(batches), desc=f"Batch Progress (Epoch {epoch+1})", unit="batch"):
				batch_loss = 0
				for i in batch:
					text,label = self.train_dataset[i]
					text  = text.to(config.DEVICE)
					label  = label.to(config.DEVICE)
					output = self.model(text)
					
					_,idxs  = torch.max(output,0)
					
					y_true.extend(label.tolist())
					y_pred.append(int(idxs))
					
					loss = self.loss_fn(output.unsqueeze(0),label)
					batch_loss+= loss
				batch_loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				epoch_loss += batch_loss.item() / len(batch)

			print(f"Epoch: {epoch + 1}    Loss: {round(epoch_loss,3)}")
			print(classification_report(y_true=y_true,y_pred=y_pred,target_names= target_names, zero_division= 0))
			


	def eval(self):
		self.model.eval()
		with torch.no_grad():
			y_true = []
			y_pred = []

			final_loss = 0
			target_names = ["negative" , "neutral" , "positive"]
			for batch in tqdm(self.test_loader):
				inputs = batch["inputs"]
				labels = batch["labels"]
				
				inputs  = inputs.to(config.DEVICE)
				labels  = labels.to(config.DEVICE)

				outputs = self.model(inputs) 

				loss 	= self.loss_fn(outputs,labels)
				_,idxs  = torch.max(outputs,1)

				final_loss += loss.item()

				y_true.extend(labels.tolist())
				y_pred.extend(idxs.tolist())
			
			avg_loss     = final_loss / len(self.test_loader)
			
			print(f"Loss: {avg_loss} \n")
			print(classification_report(y_true=y_true,y_pred=y_pred,target_names= target_names, zero_division= 0))
			
			report_dict = metils.get_metric_report(y_pred=y_pred,y_true=y_true,target_names=target_names)

			return avg_loss,report_dict
		
	def eval_updated(self):

		self.model.eval()

		with torch.no_grad():
			y_true = []
			y_pred = []

			final_loss = 0
			target_names = ["negative" , "neutral" , "positive"]

			for i in tqdm(range(len(self.test_dataset)), desc="Evaluating", ncols=100):
				text,label = self.test_dataset[i]
				text  = text.to(config.DEVICE)
				label  = label.to(config.DEVICE)
				outputs = self.model(text)

				loss = self.loss_fn(outputs.unsqueeze(0),label)
				_,idxs  = torch.max(outputs,0)

				final_loss += loss.item()

				y_true.extend(label.tolist())
				y_pred.append(int(idxs))
			avg_loss     = final_loss / len(self.test_dataset)
			print(f"Loss: {avg_loss} \n")
			print(classification_report(y_true=y_true,y_pred=y_pred,target_names= target_names, zero_division= 0))
			
			report_dict = metils.get_metric_report(y_pred=y_pred,y_true=y_true,target_names=target_names)

			return avg_loss,report_dict

