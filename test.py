import numpy as np
from evaluation import BinaryClassEvaluator, RegressionEvaluator
from sklearn.linear_model import LogisticRegression, LinearRegression

sample_size = 10000000
np.random.seed(42)

# classification
y = np.random.choice([0, 1], sample_size, p = [0.7, 0.3])
x = y + np.random.randn(sample_size)
x = x.reshape(-1, 1)

clf = LogisticRegression()
clf.fit(x, y)
yfit = clf.predict_proba(x)[:, 1]

e = BinaryClassEvaluator(y, yfit)
# print(e.segment())
e.report()
e.record(filename='test_record.txt')
# e.p_r_curve()
# e.roc_curve()
# e.stacked_hist()
# e.plot_confusion_matrix()
"""
# regression
y = np.random.randn(sample_size)
x = y + np.random.randn(sample_size)
x = x.reshape(-1, 1)

m = LinearRegression()
m.fit(x, y)
yfit = m.predict(x)

e = RegressionEvaluator(y, yfit)
e.report()
"""