from setuptools import setup
from raatk.__init__ import __version__

setup(name='raatk',
    version=__version__,
    description='reduce amino acid toolkit',
    url='https://github.com/huang-sh/raatk/',
    author='huangsh',
    author_email='hsh-me@outlook.com',
    license='BSD 2-Clause',
    packages=['raatk'],
    install_requires=[
        'numpy==1.18.2',
        'matplotlib==3.2.1',
        'scikit-learn==0.22.1',
        'seaborn==0.10.0',
        'plotly==4.8.2'
        ],
    entry_points={
        'console_scripts': [
        'raatk=raatk.__main__:command_parser',
            ]
        },
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True)
