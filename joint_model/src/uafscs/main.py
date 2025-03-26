import utils.data_utils as dutils
import utils.train_utils as tutils
import utils.console_utils as cutils
import utils.model_utils as mutils
from utils.console_utils import get_console_args
import torch
import pandas as pd
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grabs arguments
args = get_console_args()

model_name      = args.modelname
epochs          = args.epochs
lr              = args.lr
optimizer       = args.optimizer
activation      = args.activation
loss_fn         = args.loss_fn
dropout         = args.dropout
train_labels    = args.train_labels
test_labels     = args.val_labels
batch_size      = args.batch_size
weight_decay    = args.weight_decay


# Grabs data
imgs_dir = "../../data/processed/images"

# Setup dataset
train_dataset = dutils.UADataset(imgs_dir, train_labels)
test_dataset = dutils.UADataset(imgs_dir, test_labels)

# Setup dataloader
train_loader = DataLoader(train_dataset,batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Grab instantiated model
if model_name == "baseline":
    model = mutils.initialize_model(model_name)

# Train model
trainer = tutils.UAFSTrainer(model           = model,
                            targets          = "carts",
                            midpoints        = "yolo",
                            lr               = lr,
                            epochs           = epochs,
                            train_dataloader = train_loader,
                            test_dataloader  = test_loader,
                            optimizer        = optimizer,
                            loss_fn          = loss_fn)

trainer.train()


trainer.eval()

