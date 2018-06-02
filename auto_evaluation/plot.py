import matplotlib
import numpy as np
import base64
from io import BytesIO
import urllib

"""
Ploting results
Encoding Matplotlib figures into base64 String
"""
# TODO: Move all plotting function here from evaluation.py


def plotting(plot_func):
    '''
	A decorator to enable plotting individually for plotting functions.
	Keep ax=None for plotting functions to plot individually.
	'''
	def wrapper(self, ax=None, *args, **kw):
		show = False
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			show = True
		plot_func(self, ax, *args, **kw)
		if show:
			plt.show()
	return wrapper

"""
BinaryClassEvaluator Plottng functions
"""
@plotting
def p_r_curve(self, ax=None):
	"""
	Plot the precision-recall curve.
	"""
	precision, recall, _ = metrics.precision_recall_curve(self.Ytrue, self.Yfit)
	ax.step(recall, precision, color='b', alpha=0.8,
	         where='post')
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xticks(np.arange(0, 1.1, 0.1))
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.axhline(y=self.baseline, ls='--', alpha=0.8)
	ax.grid(alpha=0.8)
	ax.set_xlabel('Recall (Coverage)')
	ax.set_ylabel('Precision (Conversion Rate)')
	ax.set_title('Precision-Recall Curve')

@plotting
def roc_curve(self, ax=None):
	"""
	Plot the ROC curve.
	"""
	fpr, tpr, _ = metrics.roc_curve(self.Ytrue, self.Yfit)
	ax.plot(fpr, tpr, color='darkorange')
	ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
	ax.set_title('ROC Curve')
	ax.set_xlabel('false positive')
	ax.set_ylabel('true positive')

@plotting
def plot_confusion_matrix(self, ax=None, normalize=False):
	"""
	Plot the confusion matrix.
	"""
	cm = metrics.confusion_matrix(self.Ytrue, self.YfitBinary)
	if normalize:
		cm = cm / cm.sum(axis=1)[:, None]
	ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.set_title('Confusion Matrix')
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(['Negative', 'Positive'])
	ax.set_yticklabels(['Negative', 'Positive'])
	ax.set_xlabel('Predicted Label')
	ax.set_ylabel('True Label')
	ax.grid(False)
	fmt = '.2f' if normalize else 'd'
	t = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, format(cm[i, j], fmt),
				horizontalalignment="center",
				color="white" if cm[i, j] > t else "black")

@plotting
def hist(self, ax=None, bins=100):
	"""
	Plot the distribution of predictions
	"""
	ax.hist(self.Yfit, bins=bins)
	ax.axvline(x=self.threshold, ls='--', alpha=0.8)
	ax.set_title('Distribution of Model Predictions')
	ax.set_xlabel('Probability')
	ax.set_ylabel('Number of Users')
	Yfit_negative =self.Yfit[self.Ytrue == 0]
	Yfit_positve = self.Yfit[self.Ytrue == 1]
	ax.hist([Yfit_negative, Yfit_positve], bins=100, stacked=True)
	ax.legend(['Negative Label', 'Positive Label'])
	ax.axvline(x=self.threshold, ls='--', alpha=0.8)

@plotting
def stacked_hist(self, ax=None, bins=100):
	"""
	Plot the distribution of predictions, stacked on their true labels.
	"""
	Yfit_negative =self.Yfit[self.Ytrue == 0]
	Yfit_positve = self.Yfit[self.Ytrue == 1]
	ax.hist([Yfit_negative, Yfit_positve], bins=bins, stacked=True)
	ax.legend(['Negative Label', 'Positive Label'])
	ax.axvline(x=self.threshold, ls='--', alpha=0.8) # draw threshold

@plotting
def plot_threshold_trend(self, ax=None, thresholds=np.arange(0.1, 1, 0.1)):
	"""
	plot the trend of accuracy, precision, recall and f1 score on different thresholds.
	"""
	thresholds = list(thresholds)
	# thresholds.append(self.best_threshold)
	YfitBin_list = [(self.Yfit>=threshold).astype('int') for threshold in thresholds]
	accuracys = [metrics.accuracy_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
	precisions = [metrics.precision_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
	recalls = [metrics.recall_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
	f1s = [metrics.f1_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
	
	ax.plot(thresholds, accuracys)
	ax.plot(thresholds, precisions)
	ax.plot(thresholds, recalls)
	ax.plot(thresholds, f1s)
	ax.legend(['Accuracy', 'Precision', 'Recall', 'F1 score'])
	ax.set_xlabel('Threshold')
	ax.set_ylabel('Score')
	ax.set_title('Metric Scores by Threshold')

"""
RegressionEvaluator Plottng functions
"""
@plotting
def plot_response_vs_predictions(self, ax=None):
	ax.scatter(self.Yfit, self.Ytrue, s=5)
	ax.set_title('Response vs Predictions')
	ax.set_xlabel('Predictions')
	ax.set_ylabel('Response')
	
@plotting
def plot_e_vs_predictions(self, ax=None):
	ax.scatter(self.Yfit, self.e, s=5)
	ax.set_title('Residuals vs Predictions')
	ax.set_xlabel('Predictions')
	ax.set_ylabel('Residuals')
	ax.axhline(y=0, ls='--', alpha=0.8)

@plotting
def qq_plot(self, ax=None):
	osm, osr = sp.stats.probplot(self.e, fit=False)
	ax.scatter(osm, osr, s=5)
	ax.plot([-4, 4], [-4, 4], color='navy', linestyle='--')
	ax.set_title('QQ Plot')

@plotting
def plot_distributions(self, ax=None):
	ax.hist(self.Ytrue, bins=50, density=True, alpha=0.5)
	ax.hist(self.Yfit, bins=50, density=True, alpha=0.5)
	ax.legend(['Response', 'Prediction'])
	ax.set_title('Distributions')



"""
Base64 encoding
"""
def to_string(fig):
    imgdata = BytesIO()
    fig.savefig(imgdata)
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(imgdata.getvalue()))
    return result_string
