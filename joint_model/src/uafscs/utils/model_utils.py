import models.mg_model1 as mg_model1
from models.ct_models import regression_with_midpoints_pose
def initialize_model(model_name, dropout) :
    model = None

    if model_name == "baseline":
        model = mg_model1.MGClassifier(dropout)
        return model
    elif model_name == "regression_with_midpoints_pose":
        return regression_with_midpoints_pose()
