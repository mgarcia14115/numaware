import torch
#----------------------------------------------------------------------------------------
#  DEFAULT SETTINGS
#----------------------------------------------------------------------------------------

DESCRIPTION = "AI Lab Teamwork (SPRING 2024)"


DEFAULTS = {
	"lrate": 			0.1,
	"epochs":			7,
	"batch_size":		128,
	"dropout":			0.4,
	"num_labels":		3,
	"max_seq_length":   128,
	"num_layers":       1,
    "optimizer":		"adam",
    "loss_fn":			"mse",
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
