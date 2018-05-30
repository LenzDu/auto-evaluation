"""Generating Reports"""
import pandas as pd
import tp


def to_html(stats):
    return tp.template('base').render(values=stats)
