import torch
from uafscs.configs import defaults as config


class UARnn(torch.nn.Module):

	def __init__(self , 
			     vocab_size                   = None, 
				 embedding_size               = None, 
				 hidden_size                  = None,
				 classes                      = None, 
				 num_layers                   = None, 
				 device                       = None, 
				 dropout                      = None, 
				 **kwargs):

		super(UARnn,self).__init__()

		

		self.embeddings = torch.nn.Embedding(vocab_size , embedding_size)
		#self.rnn       	= torch.nn.GRU(embedding_size , num_layers = num_layers , hidden_size = hidden_size ,batch_first = True, device = device, dropout = dropout)
		self.rnn       	= torch.nn.GRU(embedding_size , num_layers = num_layers , hidden_size = hidden_size , device = device, dropout = dropout)
		self.fc1 		= torch.nn.Linear(hidden_size , classes ,device = device)
		#self.fc2        = torch.nn.Linear(hidden_size,classes)
		self.relu       = torch.nn.ReLU()

	def forward(self , input = None , **kwargs):

		embeddings      = self.embeddings(input)
		#output,_        = self.rnn(embeddings)
		#output          = output[:,-1,:]
		outputs,h_n     = self.rnn(embeddings)
		output          = h_n[-1]
		output          = self.relu(output)
		output          = self.fc1(output)
		
		#output          = self.fc2(output)

		return output


	