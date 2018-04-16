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

## Requirements

Scoring functionality requires only packages available in most base Python installs. However, jinja2 is required for the HTML visualization. The Pattern package -- only available for Python 2.x -- allows for additional text cleaning, potentially improving the output.

See requirements.txt for complete details.


## TF-IDF Calculation

This package uses a modified version of the standard TF-IDF equation.

Term frequency is the raw count of word w in time unit t:

    f_w,t = count(f_t)

Inverse document frequency is defined as one plus the log of the number of time units N divided by the number of time units w appears plus one:

    if_w = 1 + log( (N / (count(w_T) + 1)) )

As usual, w's score in time t is:

    TFIDF_w,t = f_w,t * if_w


## References

Viegas, Fernanda B., Scott Golder, and Judith Donath. "Visualizing Email Content: Portraying Relationships from Conversational Histories." _Proceedings of ACM CHI_, April 2006.

Manning, Christopher D., Prabhakar Raghavan, and Hinrich Schütze. _Introduction to Information Retrieval_. Cambridge UP: 2008. Ch. 6, but esp. p. 128.