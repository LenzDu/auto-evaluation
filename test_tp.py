from jinja2 import Environment, PackageLoader

pl = PackageLoader('auto_evaluation', 'templates')
env = Environment(lstrip_blocks=True, trim_blocks=True, loader=pl)
template = env.get_template('test.j2')
content = template.render(sentence="WWWorld!")


print(content)