from sklearn.metrics import classification_report
import pandas as pd

def get_metric_report(y_pred = None , y_true = None , target_names = None , **kwargs):

	
	report = classification_report(y_pred = y_pred, y_true = y_true , target_names = target_names, output_dict=True , zero_division= 0)

	#return report['macro avg']['precision'], report['macro avg']['recall'],report['macro avg']['f1-score'],report['accuracy']
	return report

def save_report(model_name      = None, 
				eval_loss       = None, 
				macro_precision = None,
				macro_recall    = None,
				macro_f1score   = None,
				accuracy        = None,
				epochs          = None,
				lr              = None,
				dropout         = None,
				num_layers      = None,
				hidden_size     = None,
				embedding_size  = None,
				max_seq_length  = None,
				optimizer       = None,
				loss_fn         = None,
				report_path     = None
				):
	
	df = pd.read_csv(report_path)
	try:
		idx = df.index[-1] + 1
	except:
		print(f"Index sixe is 0")
		idx = 0
	df.loc[idx] = [model_name,eval_loss,macro_precision,macro_recall,macro_f1score,accuracy,epochs,lr,dropout,num_layers,hidden_size,embedding_size,max_seq_length,optimizer,loss_fn]
	df.to_csv(report_path, index=False)