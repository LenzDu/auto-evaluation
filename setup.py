from setuptools import setup, find_packages

with open('requirements.txt') as fp:
  install_requires = fp.read()

setup(name='auto-evaluation',
  version='0.1',
  description='Auto Evalution',
  url='https://github.com/LenzDu/auto-evaluation',
  author='Lingzhi, Shuwen',
  author_email='vrtjso@gmail.com',
  packages=find_packages(),
  install_requires=install_requires,
  python_requires='>=3.5',
  zip_safe=False)