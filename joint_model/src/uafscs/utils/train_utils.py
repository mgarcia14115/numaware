from tqdm import tqdm

class UAFSTrainer:

    def __init__(self,
                 model              = None,
                 targets            = None,
                 midpoints          = None,
                 lr                 = None,
                 epochs             = None,
                 train_dataloader   = None,
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
        self.optimizer          = optimizer
        self.weight_decay       = weight_decay
        self.dropout            = dropout
        self.loss_fn            = loss_fn

    def train(self):

        self.model.train()


        for epoch in range(self.epochs):

            for batch in tqdm(self.train_dataloader):

                img           = batch[0]
                midpoints     = None
                targets       = None
                if self.midpoints == "yolo":
                    midpoints = batch[1]
                else: 
                    midpoints  = batch[2]
                
                if self.targets == "joints":
                    targets = batch[3]
                else:
                    targets = batch[4]
                
                self.optimizer.zero_grad()

                predictions = self.model(img,midpoints)

                loss        = self.loss_fn(predictions,targets)

                loss.backward()

                self.optimizer.step()

                
                
            print(f"Epoch: {epoch + 1}   Loss: {loss.item()}")
            
