from uafscs.utils.console_utils import get_console_args
from uafscs.utils.model_utils import initialize_model
import uafscs.utils.batch_utils	    as butils
import uafscs.utils.metrics_utils	as metils
import uafscs.utils.console_utils	as cutils
import uafscs.utils.data_utils		as dutils
import uafscs.utils.model_utils		as mutils
import uafscs.utils.train_utils		as tutils
import uafscs.models.baseline       as baseline_model
import uafscs.models.experiment1    as experiment1_model
from uafscs.configs import defaults as config
import pandas as pd
from torch.utils.data import DataLoader
import torch
import datetime
from transformers import BertTokenizer

device = config.DEVICE

args = get_console_args()
config.DEFAULTS["max_seq_length"] = args.max_seq_length

model = initialize_model(args.modelname,args.num_labels,args.embedding_size,args.hidden_size,args.vocab_size,args.num_layers,args.dropout)


splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["test"])

df_train["label"] = df_train["label"].apply(dutils.convert_labels)
df_test["label"] = df_test["label"].apply(dutils.convert_labels)

# train_dataset = dutils.CustomDataset(df_train)
# test_dataset  = dutils.CustomDataset(df_test)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = dutils.myDataset(df_train,tokenizer)
test_dataset  = dutils.myDataset(df_test,tokenizer)

custom_collate_fn = butils.rnn_collate_fn

#train_loader = DataLoader(train_dataset,batch_size=128,collate_fn=custom_collate_fn,shuffle=False)
#test_loader  = DataLoader(test_dataset ,batch_size=128,collate_fn=custom_collate_fn,shuffle=False)



optimizer = args.optimizer

if optimizer == "Adam":
	otpimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
elif optimizer == "SGD":
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
else:
	optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

loss_fn = args.loss_fn

if loss_fn == "MSE":
	loss_fn = torch.nn.MSELoss()
else:
	loss_fn = torch.nn.CrossEntropyLoss()


UATrainer = tutils.UAFSTrainer(	model        = model.to(device),
							   	optimizer    = optimizer,
								test_dataset = test_dataset,
								train_dataset= train_dataset,
								loss_fn      = loss_fn,
								epochs       = args.epochs,
								device       = device
							   )


print(f"\nStarting Training.................")
UATrainer.train_updated(batch_size=64)


print(f"\nStarting Evaluation........................")
avg_loss,report = UATrainer.eval_updated()

metils.save_report(args.modelname,
				   avg_loss,
				   report['macro avg']['precision'],
				   report['macro avg']['recall'],
				   report['macro avg']['f1-score'],
				   report['accuracy'],
				   args.epochs,
				   args.lr,
				   args.dropout,
				   args.num_layers,
				   args.hidden_size,
				   args.embedding_size,
				   args.max_seq_length,
				   args.optimizer,
				   args.loss_fn,
				   "../../reports/report.csv")
current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
weight_path = "../../local/models/"
torch.save(model,weight_path+time_str+".pth")


