import torch

class FeedForwardNN(torch.nn.Module):
	
	def __init__ (self, classes = None, hidden_size = None , vocab_size = None , embedding_dim = None , **kwargs):
		super(FeedForwardNN,self).__init__()
		
		self.embeddings = torch.nn.Embedding(num_embeddings = vocab_size , embedding_dim = embedding_dim)
		self.fc1  		= torch.nn.Linear(embedding_dim  , hidden_size)
		self.fc2  		= torch.nn.Linear(hidden_size , hidden_size)
		self.fc3  		= torch.nn.Linear(hidden_size , classes)
		
		self.relu 		= torch.nn.ReLU() 

	def forward (self , input = None , **kwargs):
		
		embeddings = self.embeddings(input)
	
		output = self.fc1(embeddings)
		output = self.fc2(output)
		output = self.fc3(self.relu(output))

		return output



	