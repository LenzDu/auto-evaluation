import numpy as np
from evaluation import BinaryClassEvaluator
from sklearn.linear_model import LogisticRegression

sample_size = 10000
y = np.random.choice([0, 1], sample_size, p = [0.6, 0.4])
x = y + np.random.randn(sample_size)
x = x.reshape(-1, 1)

clf = LogisticRegression()
clf.fit(x, y)
yfit = clf.predict_proba(x)[:, 1]

e = BinaryClassEvaluator(y, yfit, 0.5)
e.report()
e.record(filename='test_record.txt')
# e.p_r_curve()
# e.roc_curve()
# e.stacked_hist()
# e.plot_confusion_matrix()