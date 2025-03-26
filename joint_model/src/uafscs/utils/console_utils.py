import argparse
from configs import defaults as config



def get_console_args():
	parser = argparse.ArgumentParser(prog="AI Models",description="Sentiment Anaylsis Training")


	parser.add_argument("--modelname",      type=str ,   	required=True, help="This is the modelname you want to do")
	parser.add_argument("--epochs",         type=int , 		default=config.DEFAULTS["epochs"], help="Number of itterations through dataset")
	parser.add_argument("--lr",             type=float, 	default=config.DEFAULTS["lrate"], help="Learning rate for model")
	parser.add_argument("--optimizer",      type=str ,		default=config.DEFAULTS["optimizer"], choices=["adam","sgd","adamw"], help="Optimizer you want")
	parser.add_argument("--activation",     type=str ,  	choices=["Sigmoid","Relu","LRelu"], help="Which activation functions to be used")
	parser.add_argument("--loss_fn",        type=str ,      default=config.DEFAULTS["loss_fn"], choices=["crossentropy", "mse"],	help="Loss function you are choosing")
	parser.add_argument("--dropout",        type=float ,    default=config.DEFAULTS["dropout"], help="Dropout for nueral net")
	parser.add_argument("--train_labels",   type=str ,      required=True, help="The filepath to the label csv")
	parser.add_argument("--val_labels",     type=str ,      required=True, help="The filepath to the validation label csv")
	parser.add_argument("--batch_size",     type=int ,      default=config.DEFAULTS["batch_size"], help="The amount of images per batch (range 16 - 64)")
	parser.add_argument("--weight_decay",   type=float ,    help="A regularization technique used to prevent over fitting (range 0.0 - 1.0)")
	parser.add_argument("--target",			type=str,		required=True, choices=["joints", "cartesians"], help="Choose between joints or cartesians for your model")
	parser.add_argument("--midpoint",		type=str,		required=True, choices=["yolo", "ours"], help="Choose between yolo our our midpoints")

	
	kwargs = parser.parse_args()



	return kwargs