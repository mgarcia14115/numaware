
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