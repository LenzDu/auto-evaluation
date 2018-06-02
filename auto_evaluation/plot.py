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

def to_string(fig):
    imgdata = BytesIO()
    fig.savefig(imgdata)
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(imgdata.getvalue()))
    return result_string
