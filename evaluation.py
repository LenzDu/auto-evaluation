import numpy as np
import pandas as pd
import scipy as sp
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
import warnings

from reports import to_html

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

def check_type(Y):
	if not isinstance(Y, np.ndarray):
		if isinstance(Y, list):
			Y = np.array(Y)
		else:
			raise TypeError('Ytrue and Yfit should be numpy array or list')
	return Y

class BaseEvaluator(object):
	'''Base class of all evaluators'''
	def __init__(self):
		pass

	def fit(self):
		pass

	def report(self):
		print(self.summary())
		self.aggregate_plots()

	def aggregate_plots(self):
		pass

	def summary(self):
		pass
	
	def record(self, filename, note='', append=True):
		"""
		record the performance into the log file
		"""
		f = open(filename, 'a') if append else  open(filename, 'w')
		
		f.write('----------------------------------------------------\n')
		f.write('Record Time: ' + str(datetime.now()) + '\n')
		# f.write('Training Label: '+training_label+', Valid Label: '+valid_label+', model: '+model+'\n')
		f.write(self.summary())
		f.write('Note:' + note + '\n')
		f.close()
		print('Evaluation successfully saved in file "%s"' % filename)

	
class BinaryClassEvaluator(BaseEvaluator):
	def __init__(self, Ytrue, Yfit, threshold=0.5):
		self.fit(Ytrue, Yfit, threshold=threshold)
		self.get_stats()
		self.html = to_html(self.stats)
		print(self.html)

	def fit(self, Ytrue, Yfit, threshold=0.5):
		"""
		Fit the input data
		"""
		# error handling
		Ytrue = check_type(Ytrue)
		Yfit = check_type(Yfit)

		if Ytrue.ndim != 1 or Yfit.ndim != 1:
			raise ValueError('Dimension of Ytrue and Yfit should be 1')
		if len(Ytrue) != len(Yfit):
			raise ValueError('Length of Ytrue and Yfit should be equal')
		if Yfit.min() < 0 or Yfit.max() > 1:
			raise ValueError('Values in Yfit should be between 0 and 1')
		if set(Ytrue) != {0, 1}:
			raise ValueError('Values in Ytrue should be 0 or 1')
		if Yfit[0] in {0, 1} and set(Yfit) == {0, 1}:  # first condition to save time for most cases
			warnings.warn('Yfit should be soft predictions (probabilities) rather than hard predictions (0 and 1). \
						   Most metrics and visualizations are not meaningful with hard predictions.')

		# fit data
		self.Ytrue = Ytrue
		self.Yfit = Yfit
		self.baseline = Ytrue.mean()
		self.best_threshold = np.percentile(Yfit, 100-(self.baseline*100)) # TODO: find best thresholds
		if threshold == 'best':
			self.threshold = self.best_threshold
		else:
			self.threshold = threshold
		self.YfitBinary = (Yfit>=self.threshold).astype('int')

	def report(self):
		"""
		Provide a report for the evaluation
		"""
		print(self.summary())
		# print(self.get_thresholds_table())
		self.aggregate_plots()
		
	def set_threshold(self, threshold):
		"""
		Reset threshold.
		"""
		if threshold == 'best':
			self.threshold = self.best_threshold
		else:
			self.threshold = threshold
		self.YfitBinary = (Yfit>=self.threshold).astype('int')

	def aggregate_plots(self):
		"""
		Aggregate all plots together.
		"""
		# TODO: sample if dataset is too large
		plt.style.use('seaborn')
		fig, axes = plt.subplots(2, 2, figsize=(10, 10))
		self.p_r_curve(axes.flat[0])
		self.roc_curve(axes.flat[1])
		self.stacked_hist(axes.flat[2])
		# self.plot_confusion_matrix(axes.flat[3], normalize=True)
		self.plot_threshold_trend(axes.flat[3])
		plt.show();

	@plotting
	def p_r_curve(self, ax=None):
		"""
		Plot the precision-recall curve.
		"""
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

	def segment(self, n_segments=4, proportions=[0.4, 0.3, 0.25, 0.05], return_cutpoints=False, is_valid=True):
		"""
		Return a segmentation of the predictions and metrics for each segment

		Returns
		-------
		segment_df : DataFrame
		cut_points : list
		"""
		# TODO: auto segmentation
		# TODO: change to percentage

		if sum(proportions) != 1:
			raise ValueError('Proportions should add up to 1')

		cum_proportion = np.array(proportions).cumsum()
		cut_points = [np.percentile(self.Yfit, 100*i) for i in cum_proportion]
		groups = []
		for i in range(n_segments):
			if i == 0:
				groups.append(self.Yfit<=cut_points[0])
			else:
				groups.append((self.Yfit>=cut_points[i-1]) & (self.Yfit<=cut_points[i]))

		segment_df = pd.DataFrame()
		segment_df['Segment'] = range(n_segments)
		segment_df['Volume'] = [group.sum() for group in groups]
		segment_df['Proportion'] = np.round(segment_df['Volume']/segment_df['Volume'].sum(), 4)
		if is_valid:
			segment_df['# of True Labels'] = [self.Ytrue[group].sum() for group in groups]
			segment_df['Precision'] = np.round(segment_df['# of True Labels']/segment_df['Volume'], 4)
			segment_df['Coverage (Recall)'] = np.round(segment_df['# of True Labels']/segment_df['# of True Labels'].sum(), 4)

		if return_cutpoints:
			return segment_df, cut_points
		return segment_df

	def get_thresholds_table(self, thresholds=np.arange(0.1, 1, 0.1)):
		"""
		Return a table of accuracy, precision, recall and f1 score on different thresholds.
		"""
		thresholds = list(thresholds)
		thresholds.append(self.best_threshold)
		YfitBin_list = [(self.Yfit>=threshold).astype('int') for threshold in thresholds]
		accuracys = [metrics.accuracy_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
		precisions = [metrics.precision_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
		recalls = [metrics.recall_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]
		f1s = [metrics.f1_score(self.Ytrue, YfitBin) for YfitBin in YfitBin_list]

		thresholds_table = pd.DataFrame({'Threshold': thresholds, 'Accuracy': accuracys, 'Precision': precisions,
										 'Recall': recalls, 'F1 Score': f1s})

		return thresholds_table

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

	def summary(self):
		"""
		Return most common metrics and other summary information for the evaluation

		Returns:
		-------
		string
		"""
		summary_str = ''
		summary_str += 'Precision: %.3f \n' % metrics.precision_score(self.Ytrue, self.YfitBinary)
		summary_str += 'Recall: %.3f \n' % metrics.recall_score(self.Ytrue, self.YfitBinary)
		summary_str += 'F1 score: %.3f \n' % metrics.f1_score(self.Ytrue, self.YfitBinary)
		summary_str += 'Accuracy: %.3f \n' % metrics.accuracy_score(self.Ytrue, self.YfitBinary)
		summary_str += 'ROC AUC: %.3f \n' % metrics.roc_auc_score(self.Ytrue, self.Yfit)
		summary_str += 'Log Loss: %.3f \n' % metrics.log_loss(self.Ytrue, self.Yfit)
		summary_str += 'Proportion of Positive Labels: %.3f \n' % self.Ytrue.mean()
		summary_str += 'Number of Observations: %d \n' % self.Ytrue.shape[0]
		summary_str += 'Threshold %.3f \n' % self.threshold
		
		return summary_str

	def get_stats(self):
		stats = {}
		stats['precision'] = metrics.precision_score(self.Ytrue, self.YfitBinary)
		stats['recall'] = metrics.recall_score(self.Ytrue, self.YfitBinary)
		stats['f1'] = metrics.f1_score(self.Ytrue, self.YfitBinary)
		stats['accuracy'] = metrics.accuracy_score(self.Ytrue, self.YfitBinary)
		stats['rocauc'] = metrics.roc_auc_score(self.Ytrue, self.YfitBinary)
		stats['logloss'] = metrics.log_loss(self.Ytrue, self.YfitBinary)
		stats['proportion'] = self.Ytrue.mean()
		stats['n'] = self.Ytrue.shape[0]
		stats['threshold'] = self.threshold
		self.stats = stats


class MultiClassEvaluator(BaseEvaluator):
	def __init__(self, Ytrue, Yfit, class_names=None):
		self.fit(Ytrue, Yfit, class_names=class_names)

	def fit(self, Ytrue, Yfit, class_names=None):
		self.Ytrue = Ytrue
		self.Yfit = Yfit
		self.n_class = Yfit.shape[1]
		self.class_names = class_names
		self.YfitBinary = Yfit.argmax(axis=1)

	def summary(self):
		summary_str = ''
		summary_str += 'Accuracy: %.3f \n' % metrics.accuracy_score(self.Ytrue, self.YfitBinary)
		summary_str += 'Log Loss: %.3f \n' % metrics.log_loss(self.Ytrue, self.Yfit)
		summary_str += 'Proportion of Labels: %.3f \n' % self.Ytrue.mean()
		summary_str += 'Number of Observations: %d \n' % self.Ytrue.shape[0]
		return summary_str

class RegressionEvaluator(BaseEvaluator):
	def __init__(self, Ytrue, Yfit):
		self.fit(Ytrue, Yfit)

	def fit(self, Ytrue, Yfit):
		# error handling
		Ytrue = check_type(Ytrue)
		Yfit = check_type(Yfit)

		if Ytrue.ndim != 1 or Yfit.ndim != 1:
			raise ValueError('Dimension of Ytrue and Yfit should be 1')
		if len(Ytrue) != len(Yfit):
			raise ValueError('Length of Ytrue and Yfit should be equal')

		self.Ytrue = Ytrue
		self.Yfit = Yfit
		self.e = self.Ytrue - self.Yfit
		
	def summary(self):
		summary_str = ''
		summary_str += 'MSE: %.3f \n' % metrics.mean_squared_error(self.Ytrue, self.Yfit)
		# summary_str += 'SSE: %.3f \n' % sum((self.Ytrue - self.Ytrue.mean()) ** 2)
		summary_str += 'R^2: %.3f \n' % metrics.r2_score(self.Ytrue, self.Yfit)
		summary_str += 'MAE: %.3f \n' % metrics.mean_absolute_error(self.Ytrue, self.Yfit)
		summary_str += 'Response Mean: %.3f \n' % self.Ytrue.mean()
		summary_str += 'Number of Observations: %d \n' % self.Ytrue.shape[0]

		return summary_str

	def aggregate_plots(self):
		plt.style.use('seaborn')
		fig, axes = plt.subplots(2, 2, figsize=(10, 10))
		self.plot_response_vs_predictions(axes.flat[0])
		self.plot_e_vs_predictions(axes.flat[1])
		self.qq_plot(axes.flat[2])
		self.plot_distributions(axes.flat[3])
		plt.show();	

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