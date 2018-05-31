"""Generating Reports"""
import pandas as pd
import auto_evaluation.templates as templates


def to_html(stats):
    return templates.template('base').render(values=stats)
