#!/usr/bin/env python

import os
import re

FILE = os.path.dirname(__file__)
STOPWORDS = set(map(str.strip, open(os.path.join(FILE, 'stopwords')).readlines()))

class TempoTFIDF(object):
    """ Python object for generating term weight for a set of documents
    produced over time, i.e., temporal TF-IDF scores.

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
        The time unit to compare term weight over. Accepts day, month, or year.

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

    def __init__(self, documents, dates, date_format='%Y-%m-%d',
                 time_unit='month', stopwords=None, max_font_size=None,
                 word_regexp=None, colocations=False):
        self.documents = documents
        self.dates = dates
        self.date_format = date_format
        self.time_unit = time_unit
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.max_font_size = max_font_size
        self.word_regexp = word_regexp if word_regexp is not None
                                                            else r'\w[\w']+'
        self.collocations = collocations

    def score_documents(self):
        """ Clean document text and produce term weights.

        Returns
        -------
        dict of dicts like:

            {'document1':
                {'taste': 5.9808,
                 'better': 3.34},
             'document2':
                {'wrong': 4.343,
                 'two': 1.34}}
        """
        clean_docs = [self.process_text(doc) for doc in self.documents]
        frequencies = self.calculate_frequencies(clean_docs)
        return self.generate_from_frequencies(frequencies)

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
        pass

    def calculate_frequencies(self, documents):
        """
        """
        pass

    def generate_from_frequencies(frequencies):
        """
        """
        pass
