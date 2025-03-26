import models.mg_model1 as mg_model1

def initialize_model(model_name, dropout) :
    model = None

    if model_name == "baseline":
        model = mg_model1.MGClassifier(dropout)
        return model
    
