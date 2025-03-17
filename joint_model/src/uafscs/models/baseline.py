import torch

class FeedForwardNN(torch.nn.Module):
	
	def __init__ (self, hidden_size = None, num_layers = None, dropout = False, dropout_num = None, activation = None , **kwargs):
		super(FeedForwardNN,self).__init__()

		
		self.seq = torch.nn.Sequential(
			torch.nn.Linear(2, hidden_size)
		)

		if activation:
			act = None
			if activation == 'Sigmoid':
				act = torch.nn.Sigmoid()
			elif activation == 'Relu':
				act = torch.nn.ReLU()
			else:
				act = torch.nn.LeakyReLU()
			for i in range(num_layers - 2):
				self.seq = torch.nn.Sequential(
					self.seq,
					act,
					
					torch.nn.Linear(hidden_size, hidden_size)
				)
			self.seq = torch.nn.Sequential(
				self.seq,
				act,
				torch.nn.Linear(hidden_size, 6)
			)
		else:
			for i in range(num_layers - 2):
				self.seq = torch.nn.Sequential(
					self.seq,
					torch.nn.Linear(hidden_size, hidden_size)
				)
			self.seq = torch.nn.Sequential(
				self.seq,
				torch.nn.Linear(hidden_size, 6)
			)




		
		

	def forward (self , input = None , **kwargs):

		return self.seq(input)



	