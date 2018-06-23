import matplotlib
import numpy as np
import base64
from io import BytesIO
import urllib
from sklearn import metrics


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


@plotting
def p_r_curve(self, ax=None):
    """
    Plot the precision-recall curve.
    """
    precision, recall, _ = metrics.precision_recall_curve(
        self.Ytrue, self.Yfit)
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


def to_string(fig):
    imgdata = BytesIO()
    fig.savefig(imgdata)
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + \
        urllib.parse.quote(base64.b64encode(imgdata.getvalue()))
    return result_string
