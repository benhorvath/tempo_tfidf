# Temporal TF-IDF Term-Document Scores
Provides a Python class to easily calculate the TF-IDF scores of a collection of documents produced over time. The class can also create an HTML page visualizing how documents' topics vary over time.

## Example

The file example.py shows a small example based on Kenneth Lay's sent messages e-mails, taken from the [Enron e-mail archive](https://www.cs.cmu.edu/~enron/).

The usage is fairly simple:

```{python}
from tempo_tfidf import TempoTFIDF

dates = ['2007-05-23', '2008-04-23']
docs = ['This is one doc', 'This is the doc with a strange word']

scorer = TempoTFIDF()
doc_scores = scorer.score_documents(docs, dates, time_unit='month')
scorer.visualize(doc_scores, path='visualize.html')
```

## References

Viegas, Fernanda B., Scott Golder, and Judith Donath. "Visualizing Email Content: Portraying Relationships from Conversational Histories." _Proceedings of ACM CHI_, April 2006.
