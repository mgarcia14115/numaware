
# Command to train a model

30522 is the size of the BertTokenizer.from_pretrained("bert-base-uncased") tokenizer.


This command has all the command line args filled.
```
python train.py --modelname experiment1 --epochs 7 --lr 0.01 --optimizer AdamW --hidden_size 10 --vocab_size 30522 --embedding_size 400 --max_seq_length 128 --loss_fn CrossEntropy --num_labels 3 --num_layers 2 --dropout .4
```

This command has the necessary ones filled.  The rest will be filled with the commands in the configs folder.

```
python train.py --modelname foo --optimizer AdamW --hidden_size 10 --vocab_size 30522 --embedding_size 300 --loss_fn CrossEntropy 
```
## Setup Project
The following steps show how to set up this project. The assumption is that you are running linux. 


1. Clone the project and install requirements
```
git clone git@github.com:mgarcia14115/numaware.git 
```

2. Install python and create a virtual enviroment
```
sudo zypper in python311
```

```
python3.11 -m venv .venv
```

```
source .venv/bin/activate
```

3. Install requirements

```
pip install --upgrade pip 
```

```
pip install -r requirements.txt 
```


## Training Our Joint Model
1. Make sure you are in the **numaware/joint_model/src/uafscs/** directory
2. The below command is only the **required** command line arguments. Type **python3.11 main.py -h** to view the optional commands.

```
python3.11 main.py --modelname=baseline --train_labels=../../data/processed/train_data.csv --val_labels=../../data/processed/test_data.csv --target=cartesians --midpoint=yolo
```
