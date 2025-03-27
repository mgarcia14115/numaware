from tqdm            import tqdm
from sklearn.metrics import r2_score
import torch
import utils.metric_utils		as meutils
from configs import defaults as config

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
                 loss_fn            = None
                    ):
        
        self.model              = model
        self.targets            = targets
        self.midpoints          = midpoints
        self.lr                 = lr
        self.epochs             = epochs
        self.train_dataloader   = train_dataloader
        self.test_dataloader    = test_dataloader
        self.weight_decay       = weight_decay
       

        
        if optimizer.lower() == "adamw":
            self.optimizer   = torch.optim.AdamW(model.parameters(),lr=self.lr)
        elif optimizer.lower() == "adam":
            self.optimizer   = torch.optim.Adam(model.parameters(),lr=self.lr)
        else:
            self.optimizer   = torch.optim.SGD(model.parameters(),lr=self.lr)

        
        if loss_fn.lower()  == "mse":
            self.loss_fn = torch.nn.MSELoss()
        if loss_fn.lower()  == "crossentropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        


    def train(self):

        self.model.train()

        
        for epoch in range(self.epochs):
            y_true=[]
            y_pred=[]
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
                    targets = batch[4]        # Use Cartesians as targets

                img = img.to(config.DEVICE)
                midpoints = midpoints.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
				
                self.optimizer.zero_grad()

                predictions = self.model(img,midpoints)

                loss        = self.loss_fn(predictions,targets)

                loss.backward()

                self.optimizer.step()
                y_true.extend(targets.detach())
                y_pred.extend(predictions.detach())
				
                
                
            print(f"Epoch: {epoch + 1}   Training Loss: {round(float(loss.item()),4)}  Training R2 score: {r2_score(y_true.cpu(),y_pred.cpu())} ")
            
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

                img = img.to(config.DEVICE)
                midpoints = midpoints.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                predictions = self.model(img,midpoints)

                loss        = self.loss_fn(predictions,targets)
                y_true.extend(targets.detach())
                y_pred.extend(predictions.detach())
                
            meutils.save_report(self.model, loss, self.epochs, self.lr, self.optimizer, self.loss_fn, r2_score(y_true, y_pred), "./model_metrics.csv" )
            print(f"Testing Loss: {round(float(loss.item()),4)}  Testing R2 score: {r2_score(y_true.cpu(),y_pred.cpu())} ")
