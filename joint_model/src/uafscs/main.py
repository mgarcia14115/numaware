import utils.data_utils as dutils
import utils.train_utils as tutils
import utils.console_utils as cutils
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
train_data = "../../data/processed/train_data.csv"
test_data = "../../data/processed/test_data.csv"

# Setup dataset
train_dataset = dutils.UADataset(imgs_dir, train_data)
test_dataset = dutils.UADataset(imgs_dir, test_data)

# Setup dataloader
train_loader = DataLoader(train_dataset,batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



