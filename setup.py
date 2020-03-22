from setuptools import setup

setup(name='raatk',
    version='0.1',
    description='reduce amino acid toolkit',
    url='https://github.com/huang-sh/raatk/',
    author='huangsh',
    author_email='hsh-me@outlook.com',
    license='MIT',
    packages=['raatk'],
    install_requires=[
        'numpy>=1.16.2',
        'matplotlib>=3.1.1',
        'scikit-learn>=0.22.1',
        'seaborn>=0.9.0',
        
        ],
    entry_points={
        'console_scripts': [
        'raa=raatk.__main__:command_parser',
            ]
        },
    python_requires=">=3.5",
    include_package_data=True,
    zip_safe=True)
