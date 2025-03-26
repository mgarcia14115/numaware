import models.mg_model1 as mg_model1

def initialize_model(model_name) :
    model = None

    if model_name == "baseline":
        model = mg_model1.MGClassifier(0.4)
        return model
    
