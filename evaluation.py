import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from datetime import datetime

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

class Evaluator(object):
	'''Super class of all evaluators'''
	def __init__(self):
		pass

	def fit(self):
		pass

	def report(self):
		pass

	def summary(self):
		pass
	
	def record(self, filename, note='', append=True):
		# record the performance into the log file
		if append:
			f = open(filename, 'a')
		else:
			f = open(filename, 'w')
		f.write('----------------------------------------------------\n')
		f.write('Record Time: ' + str(datetime.now()) + '\n')
		# f.write('Training Label: '+training_label+', Valid Label: '+valid_label+', model: '+model+'\n')
		f.write(self.summary())
		f.write('Note:' + note + '\n')
		f.close()
		print('Evaluation successfully saved in file "%s"' % filename)

	
class BinaryClassEvaluator(Evaluator):
	def __init__(self, Ytrue, Yfit, threshold=None, ratio=1):
		self.fit(Ytrue, Yfit, threshold, ratio)

	def fit(self, Ytrue, Yfit, threshold, ratio=1):
		self.Ytrue = Ytrue
		self.Yfit = Yfit
		self.baseline = Ytrue.mean()
		self.ratio = ratio
		if threshold is not None:
			self.threshold = threshold
		else:
			self.threshold = np.percentile(Yfit, 100-(self.baseline*self.ratio*100))
		self.YfitBinary = (Yfit>=self.threshold).astype('int')

	def report(self):
		plt.style.use('seaborn')
		print(self.summary())
		fig, axes = plt.subplots(2, 2, figsize=(12, 12))
		self.p_r_curve(axes.flat[0])
		self.roc_curve(axes.flat[1])
		self.stacked_hist(axes.flat[2])
		self.plot_confusion_matrix(axes.flat[3], normalize=True)
		plt.show();

	@plotting
	def p_r_curve(self, ax=None):
		precision, recall, _ = metrics.precision_recall_curve(self.Ytrue, self.Yfit)
		ax.step(recall, precision, color='b', alpha=0.8,
		         where='post')
		ax.set_xlim(0,1)
		ax.set_ylim(0,1)
		ax.set_xticks(np.arange(0,1.1,0.1))
		ax.set_yticks(np.arange(0,1.1,0.1))
		ax.axhline(y=self.baseline, ls='--', alpha=0.8)
		ax.grid(alpha=0.8)
		ax.set_xlabel('Recall (Coverage)')
		ax.set_ylabel('Precision (Conversion Rate)')
		ax.set_title('Precision-Recall Curve')

	@plotting
	def roc_curve(self, ax=None):
		fpr, tpr, _ = metrics.roc_curve(self.Ytrue, self.Yfit)
		ax.plot(fpr, tpr, color='darkorange')
		ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
		ax.set_title('ROC Curve')
		ax.set_xlabel('false positive')
		ax.set_ylabel('true positive')

	@plotting
	def plot_confusion_matrix(self, ax, normalize=False):
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
	def hist(self, ax=None):
		ax.hist(self.Yfit, bins=50)
		ax.axvline(x=self.threshold, ls='--', alpha=0.8)
		ax.set_title('Distribution of Model Predictions')
		ax.set_xlabel('Probability')
		ax.set_ylabel('Number of Users')

	@plotting
	def stacked_hist(self, ax=None):
		Yfit_negative =self.Yfit[self.Ytrue == 0]
		Yfit_positve = self.Yfit[self.Ytrue == 1]
		ax.hist([Yfit_negative, Yfit_positve], bins=100, stacked=True)
		ax.legend(['Negative Label', 'Positive Label'])
		ax.axvline(x=self.threshold, ls='--', alpha=0.8)

	def segment(self, n_segments=4, proportions=[0.95, 0.7, 0.3], is_valid=True, return_cutpoints=False):
		cut_points = [np.percentile(self.Yfit, 100*i) for i in proportions]
		p_group = self.Yfit>=cut_points[0]
		lp_group = (self.Yfit<cut_points[0]) & (self.Yfit>=cut_points[1])
		ln_group = (self.Yfit<cut_points[1]) & (self.Yfit>=cut_points[2])
		n_group = self.Yfit<cut_points[2]
		segment_df = pd.DataFrame()
		segment_df['segment'] = ['Positive', 'Likely positive', 'Liekly negative', 'Negative']
		segment_df['volume'] = [EA_group.sum(), highEA_group.sum(), rareEA_group.sum(), nonEA_group.sum()]
		segment_df['User Proportion'] = np.round(segment_df['volume']/segment_df['volume'].sum(), 3)
		if is_valid:
			segment_df['EA Converter'] = self.Ytrue[EA_group].sum(), self.Ytrue[highEA_group].sum(), self.Ytrue[rareEA_group].sum(), self.Ytrue[nonEA_group].sum()
			segment_df['EA Percentage'] = np.round(segment_df['EA Converter']/segment_df['EA Converter'].sum(), 4)
			segment_df['Conversion Rate'] = np.round(segment_df['EA Converter']/segment_df['volume'], 4)

		if return_cutpoints:
			return segment_df, cut_points
		return segment_df

	def summary(self):
		summary_str = ''
		summary_str += 'Precision: %.3f \n' % metrics.precision_score(self.Ytrue, self.YfitBinary)
		summary_str += 'Recall: %.3f \n' % metrics.recall_score(self.Ytrue, self.YfitBinary)
		summary_str += 'F1 score: %.3f \n' % metrics.f1_score(self.Ytrue, self.YfitBinary)
		summary_str += 'Accuracy: %.3f \n' % metrics.accuracy_score(self.Ytrue, self.YfitBinary)
		summary_str += 'ROC AUC: %.3f \n' % metrics.roc_auc_score(self.Ytrue, self.YfitBinary)
		summary_str += 'Log Loss: %.3f \n' % metrics.log_loss(self.Ytrue, self.YfitBinary)
		summary_str += 'Proportion of Positive Labels: %.3f \n' % self.Ytrue.mean()
		summary_str += 'Number of Observations: %d \n' % self.Ytrue.shape[0]
		summary_str += 'Threshold %.3f \n' % self.threshold
		
		return summary_str

class MultiClassEvaluator(Evaluator):
	def __init__(self, Ytrue, Yfit):
		self.fit(Ytrue, Yfit)

	def fit(self, Ytrue, Yfit):
		self.Ytrue = Ytrue
		self.Yfit = Yfit