from tqdm            import tqdm
from sklearn.metrics import r2_score
import torch

class UAFSTrainer:

    def __init__(self,
                 model              = None,
                 targets            = None,
                 midpoints          = None,
                 lr                 = None,
                 epochs             = None,
                 train_dataloader   = None,
                 test_dataloader    = None,
                 optimizer          = None,
                 weight_decay       = None,
                 dropout            = None,
                 loss_fn            = None
                    ):
        
        self.model              = model
        self.targets            = targets
        self.midpoints          = midpoints
        self.lr                 = lr
        self.epochs             = epochs
        self.train_dataloader   = train_dataloader
        self.test_dataloader    = test_dataloader
        self.optimizer          = optimizer
        self.weight_decay       = weight_decay
        self.dropout            = dropout
        self.loss_fn            = loss_fn

    def train(self):

        self.model.train()

        y_true=[]
        y_pred=[]
        for epoch in range(self.epochs):

            for batch in tqdm(self.train_dataloader):

                img           = batch[0]
                midpoints     = None
                targets       = None
                if self.midpoints.lower() == "yolo":  # Check if user wants to use yolo midpoints
                    midpoints = batch[1]              # Using yolo midpoints
                else: 
                    midpoints  = batch[2]             # Using our midpoints
                
                if self.targets.lower() == "joints":  # Check if user wants to user joints as targets
                    targets = batch[3]                # Use joints as targets
                else:
                    targets = batch[4]                # Use Cartesians as targets
                
                self.optimizer.zero_grad()

                predictions = self.model(img,midpoints)

                loss        = self.loss_fn(predictions,targets)

                loss.backward()

                self.optimizer.step()
                y_true.extend(targets.detach())
                y_pred.extend(predictions.detach())

                
                
            print(f"Epoch: {epoch + 1}   Training Loss: {round(float(loss.item()),4)}  Training R2 score: {r2_score(y_true,y_pred)} ")
            
    def eval(self):
        
        self.model.eval()
        y_true=[]
        y_pred=[]
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):

                img           = batch[0]
                midpoints     = None
                targets       = None
                if self.midpoints.lower() == "yolo":
                    midpoints = batch[1]
                else: 
                    midpoints  = batch[2]
                
                if self.targets.lower() == "joints":
                    targets = batch[3]
                else:
                    targets = batch[4]

                predictions = self.model(img,midpoints)

                loss        = self.loss_fn(predictions,targets)
                

            print(f"Test Loss: {loss.item()}  Test R2 score: {r2_score(y_true,y_pred)} ")