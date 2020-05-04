import setuptools
from clustermatch import __version__, __short_description__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustermatch",
    version=__version__,
    author="Milton Pividori",
    author_email="miltondp@gmail.com",
    description=__short_description__,
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sinc-lab/clustermatch",
    packages=['clustermatch', 'clustermatch/utils'],
    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'joblib',
        'scikit-learn',
        'xlrd',
        'xlwt',
        'openpyxl',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console"
    ],
    entry_points={
        'console_scripts': [
            'clustermatch = clustermatch.main:run'
        ]
    },
)
