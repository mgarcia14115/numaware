from models.ct_models import regression_with_midpoints_pose, regression_with_images_midpoints, VisionTransformer
def initialize_model(model_name, dropout) :
    if model_name == "regression_with_midpoints_pose":
        return regression_with_midpoints_pose()
    elif model_name == "regression_with_images_midpoints":
        return regression_with_images_midpoints()
    elif model_name == "trans":
        return VisionTransformer()
