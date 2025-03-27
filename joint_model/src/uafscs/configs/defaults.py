import torch
#----------------------------------------------------------------------------------------
#  DEFAULT SETTINGS
#----------------------------------------------------------------------------------------

DESCRIPTION = "AI Lab Teamwork (SPRING 2024)"


DEFAULTS = {
	"lrate": 			0.1,
	"epochs":			10,
	"batch_size":		32,
	"dropout":			0.0,	
    "optimizer":		"adam",
    "loss_fn":			"mse",
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
