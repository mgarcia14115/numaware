from uafscs.configs import defaults as config
import uafscs.models.baseline           as baseline
import uafscs.models.experiment1        as experiment1


def initialize_model(model_name, num_labels = None, embedding_size = None, hidden_size = None, vocab_size = None, num_layers = None,dropout = None,**kwargs):
	
	model = None

	if model_name == "baseline":
		return model
	elif model_name == "experiment1":
		model = experiment1.UARnn(vocab_size       = vocab_size, 
							      embedding_size   = embedding_size,
								  hidden_size      = hidden_size,
								  classes          = num_labels,
								  num_layers       = num_layers,
								  device           = config.DEVICE,
								  dropout          = dropout)
		
	else:

		return model
	
	return model

