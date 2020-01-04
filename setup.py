from setuptools import setup

setup(name='raa_assess',
    version='0.1',
    description='reduce amino acid assess',
    url='https://github.com/huang-sh/raa-assess/',
    author='huangsh',
    author_email='hsh-me@outlook.com',
    license='MIT',
    packages=['raa_assess'],
    install_requires=[
        'numpy>=1.16.2',
        'matplotlib>=3.0.3',
        'scikit-learn>=0.22.1',
        'seaborn>=0.9.0',
        
        ],
    entry_points={
        'console_scripts': [
        'raatk=raa_assess.__main__:command_parser',
            ]
        },
    python_requires=">=3.5",
    include_package_data=True,
    zip_safe=True)