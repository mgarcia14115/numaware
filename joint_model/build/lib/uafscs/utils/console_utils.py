import argparse
from uafscs.configs import defaults as config



def get_console_args():
	parser = argparse.ArgumentParser(prog="AI Models",description="Sentiment Anaylsis Training")


	parser.add_argument("--modelname",      type=str ,   	choices=["baseline","experiment1","experiment2"], help="This is the modelname you want to do")
	parser.add_argument("--epochs",         type=int , 		default=config.DEFAULTS["epochs"], help="Number of itterations through dataset")
	parser.add_argument("--lr",             type=float, 	default=config.DEFAULTS["lrate"], help="Learning rate for model")
	parser.add_argument("--optimizer",      type=str ,		choices=["Adam","SGD","AdamW"], help="Optimizer you want")
	parser.add_argument("--hidden_size",    type=int ,  	help="Hidden size")
	parser.add_argument("--vocab_size",     type=int ,  	help="Number of unique tokens")
	parser.add_argument("--embedding_size", type=int ,      help="size of the embedding")
	parser.add_argument("--max_seq_length", type=int , 		default=config.DEFAULTS["max_seq_length"], help="The sequence lenght")
	parser.add_argument("--loss_fn",        type=str ,      choices=["CrossEntropy", "MSE"],	help="Loss function you are choosing")
	parser.add_argument("--num_labels",     type=int ,    	default=config.DEFAULTS["num_labels"], help="Number of Labels in dataset")
	parser.add_argument("--num_layers",     type=int ,    	default=config.DEFAULTS["num_layers"], help="Number of layers in neural net")
	parser.add_argument("--dropout",        type=float ,    default=config.DEFAULTS["dropout"], help="Dropout for nueral net")
	
	kwargs = parser.parse_args()



	return kwargs