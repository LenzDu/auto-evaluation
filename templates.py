"""Create a jinja2 template environment in order to load templates for our application"""

from jinja2 import Environment, PackageLoader

# initializing jinja2
pl = PackageLoader('auto_evaluation', 'templates')  # the loader will look up the "templates" folder in your "auto_evaluation" package
jinja2_env = Environment(lstrip_blocks=True, trim_blocks=True, loader=pl)
# lstrip_blocks: leading spaces and tabs are stripped from the start of a line to a block.
# trim_blocks: the first newline after a block is removed (block, not variable tag!).

# Mapping between template name and file
templates = {'base':'base.html'}

def template(template_name):
    """Return an jinja2 template that is ready to be renderred
    
    Parameters
    ----------
    template_name: name of the template that specify in the template-file mapping

    Returns
    -------
    the jinja2 template
    """
    return jinja2_env.get_template(templates[temppate_name])    # get_template(): return the loaded template
