import pandas as pd

def save_report(model_name      = None, 
				eval_loss       = None, 
				epochs          = None,
				lr              = None,
				optimizer       = None,
				loss_fn         = None,
				r2_loss 		= None,
				report_path     = None
				):
	
	
	try:
		df = pd.read_csv(report_path)
		new = [model_name,epochs,lr,optimizer,loss_fn, eval_loss, r2_loss]
		df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
		df.to_csv(report_path, index=False)
	except:
		new = [model_name,epochs,lr,optimizer,loss_fn, eval_loss, r2_loss]
		df = pd.DataFrame([new], columns=['Model Architecture', 'Epochs', 'Learning Rate', 'Optimizer', 'loss_fn', 'loss', 'r2_loss'])
		df.to_csv(report_path, index=False)
	