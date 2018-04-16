#!/usr/bin/env python

from disutils.core import setup

setup(
    name='Temporal TF-IDF',
    version='1.0',
    author='Ben Horvath',
    author_email='benhorvath@gmail.com',
    packages='tempo_tfidf',
    scripts='example.py',
    url='https://github.com/benhorvath/tempo_tfidf',
    license='LICENSE',
    description='Generate TF-IDF scores for documents produced over time',
    long_description=open('README.md').read(),
    install_requires=[
        'Jijna2==2.10',
        'MarkupSafe==1.0',
        'Pattern==2.6',
    ],
)
