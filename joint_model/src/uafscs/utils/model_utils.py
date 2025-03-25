from uafscs.configs import defaults as config
import uafscs.models.baseline           as baseline
import uafscs.models.experiment1        as experiment1


def initialize_model(model_name,**kwargs):
	
	model = None

	if model_name == "baseline":
		return model
	elif model_name == "experiment1":
		model = experiment1.UARnn()
		
	else:

		return model
	
	return model

