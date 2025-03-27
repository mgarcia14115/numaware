import models.mg_model1 as mg_model1

def initialize_model(model_name, dropout) :
    model = None

    if model_name == "mg":
        model = mg_model1.RegWithMid(dropout)
        return model
    
