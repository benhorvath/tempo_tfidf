#!/usr/bin/env python

""" Short example script demonstrating usage of the TempoTFIDF object.

Pandas is not strictly necessary, but it used as simple way to access the
comma-delimited example data located in the file example_data."""

import pandas as pd

from tempo_tfidf import TempoTFIDF

df = pd.read_csv('example_data', sep=',')
docs = df['message'].tolist()
dates = df['date'].tolist()

scorer = TempoTFIDF()

doc_scores = scorer.score_documents(docs, dates, time_unit='month')
scorer.visualize(doc_scores, path='visualize.html')  # creates HTML file