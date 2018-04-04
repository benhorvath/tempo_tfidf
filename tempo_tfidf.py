#!/usr/bin/env python

""" Class definition for Temporal TF-IDF object. Provides term-weighting a la
TF-IDF but for a collection of documents produced over time, i.e., associated
with some date.

TODO
----
- Add functions to collocation=True feature.
- Visualize() function -- move viz options from init to this function
- singularize function in process_words?
- Double check documentation

"""

from collections import Counter
from datetime import datetime
import math
import os
import re

FILE = os.path.dirname(__file__)
STOPWORDS = set([w.strip() for w in open(os.path.join(FILE,
    'stopwords')).readlines()])

class TempoTFIDF(object):
    """ Python object for generating term weight for a set of documents
    produced over time, i.e.:, temporal TF-IDF scores.

    Parameters
    ----------
    stopwords : str (default=None)
        File path to custom stopwords document, with each stopword on its own
        line seperated by a newline \\n. If left blank, will use a default set
        of stopwords contained in the file 'stopwords.'

    max_font_size : int (default=100)
        Maximum font size for visualization.

    word_regexp : string (default=None)
         Regular expression defining a word for tokenization purposes.

    collocations : bool (default=False)
        Whether to include bigrams in weighting.
    """

    def __init__(self, stopwords=None, max_font_size=None, word_regexp=None,
                 collocations=False):
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.max_font_size = max_font_size
        self.word_regexp = word_regexp if word_regexp is not None else r"\w[\w']+"
        self.collocations = collocations

    def score_documents(self, documents, dates, date_format='%Y-%m-%d',
                        time_unit='month'):
        """ Clean document text and produces term weights by date.

        Parameters
        ----------
        documents : list
            Documents to be analyzed.

        dates : list
            Dates associated with each document. len(documents) must equal
            len(documents).

        date_format : str (default='%Y-%m-%d')
             Formatting of dates in dates parameter.

        time_unit : str (default='month')
            The time unit to compare term weight over. Accepts day, month, or
            year.

        Returns
        -------
        dict of dicts like:

            {'April 2018':
                {'taste': 5.9808,
                 'better': 3.34},
             'May 2018':
                {'wrong': 4.343,
                 'two': 1.34}}
        """

        # Aggregate documents according to time_unit
        aggr_dates = [self.extract_date(d, time_unit, date_format=date_format) for d in dates]
        aggr_docs = {}
        for i, doc in enumerate(documents):
            dt = aggr_dates[i]
            if dt not in aggr_docs:
                aggr_docs[dt] = []
            aggr_docs[dt].append(doc)
        time_docs = {k: ' '.join(v) for k,v in aggr_docs.items()}

        doc_tokens = {k: self.process_text(v) for k,v in time_docs.items()}
        doc_freqs = {k: self.calculate_word_frequencies(v) for k,v in doc_tokens.items()}

        # Calculate document collection frequencies
        collection_tokens = []
        for k,v in doc_tokens.items():
            for i in v:
                collection_tokens.append(i)
        collection_freqs = self.calculate_word_frequencies(collection_tokens)
        return self.generate_from_frequencies(doc_freqs, collection_freqs, aggr_dates)

    def process_text(self, document):
        """ Preprocesses a document into a clean list of tokens. Removes
        stopwords, punctuation, numbers, and attempts to normalizes plurals
        and other word endings.

        If self.collocations is set to True, will return bigrams as well as
        unigrams.

        Parameters
        ----------
        document : str
            Documents to be cleaned

        Returns
        -------
        List of tokens
        """
        stopwords = set([w.lower() for w in STOPWORDS])
        word_def = r"\w[\w']+"
        words = re.findall(word_def, document, 0)
        words = [w.lower() for w in words if w.lower() not in stopwords]
        words = [w[:2] if w.endswith("'s") else w for w in words]
        words = [w for w in words if not w.isdigit()]
        return words

    def calculate_word_frequencies(self, tokens):
        """ Counts occurances of tokens in list of tokens.

        Parameters
        ----------
        tokens : list
            List of strings (tokens) to tabulate

        Returns
        -------
        Dictionary of tokenss and corresponding counts
        """
        return dict(Counter(tokens))

    def generate_from_frequencies(self, document_frequencies,
                                  collection_frequencies, dates,
                                  date_format='%Y-%m-%d', date_unit='month'):
        """ Generates the temporal TF-IDF scores for each document-term dyad
        in the collection of documents.

        TODO: Describe the formula.

        TODO
        ----
        - Add date_unit functionality

        parameters
        ----------
        document_frequencies : list
            A list of document-term frequencies, as dicts like {'word': 2},
            each list element corresponding to an input document.

        collection_frequencies : dict
            Token frequency for the collection of documents like {'word': 2}.

        dates : list
            Dates associated with each document. len(dates) must equal
            len(documents).

        date_format : str (default='%Y-%m-%d')
             Formatting of dates in dates parameter.

        time_unit : str (default='month')
            The time unit to compare term weight over. Accepts day, month, or
            year.

        Returns
        -------
        Dict corresponding to input documents and their dates

            {'February 2015': {'taste': 5.980538, 'better': 3.3423432},
             'March 2015': {'wrong': 4.343234, 'two': 1.34234234}}
        """
        documents_scores = {}

        for k,v in document_frequencies.items():
            document_scores = dict()
            for token in v:
                f_w_t = v[token]
                if_w = math.log ( 1 / float(collection_frequencies[token]) )
                s = f_w_t * if_w
                document_scores[token] = s
            documents_scores[k] = document_scores

        self.documents_scores = documents_scores
        return documents_scores

    def visualize(self, document_scores, path='visualize.html'):
        """ Produces an HTML file to visualize change over time.

        Parameters
        ----------
        document_scores : dict
            Output of self.score_documents

        Returns
        -------
        Saves an HTML file to path
        """
        pass

    @staticmethod
    def extract_date(d, part, date_format='%Y-%m-%d'):
        """ Extracts a date part from a date as a string. Can select week, month, or year."""
        dt = datetime.strptime(d, date_format)

        if part == 'week':
            return dt.strftime('w%U %Y')
        elif part == 'month':
            return dt.strftime('%B %Y')
        elif part == 'year':
            return dt.strftime('%Y')
        else:
            msg = 'Valid date part options are week, month, and year'
            raise ValueError(msg)


if __name__ == '__main__':

    docs = ['While troubleshooting HIVE performance issues when TEZ engine is being used there may be a need to increase the number of mappers used during a query.', 'In Monday\'s damp predawn darkness, teachers gathered in front of Muskogee High. But instead of heading to their classrooms, they piled on to a bus painted with the schoo\'s mascot the Roughers and headed 150 miles west to Oklahoma City.', 'In this query, you can create dimensions from customer_id and lifetime_spend. However, suppose you wanted the user to be able to specify the region, instead of hard-coding it to "northeast". The region cannot be exposed as a dimension, and therefore the user cannot filter on it as normal.']


    dts = ['2018-01-01', '2018-01-02', '2018-02-01']

    scorer = TempoTFIDF()
    x = scorer.score_documents(docs, dts,time_unit='week')
    print(x)
    
