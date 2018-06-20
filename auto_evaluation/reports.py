"""Generating Reports"""
import pandas as pd
import auto_evaluation.templates as templates


def to_html(stats):
    overview_html=templates.template('overview').render(values=stats)
    tabs_html=templates.template('tabs').render(values=stats)
    return templates.template('base').render({
        'overview_html': overview_html,
        'tabs_html': tabs_html
    })
