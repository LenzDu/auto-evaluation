"""Generating Reports"""
import pandas as pd
import auto_evaluation.templates as templates


def to_html(sample, stats):
    
    return templates.templates('base').render({})
