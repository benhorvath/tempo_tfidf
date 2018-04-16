#!/usr/bin/env python

""" Class definition for Temporal TF-IDF object. Uses a version of term weight-
inverse document weight to score and visualize key terms in a documents
produced over time. Input is two lists of the same length, one containing the
message and the other containing an associated date.

TODO
----
- Test various kinds of documents and make small adjustments as necessary

"""

from collections import Counter
import csv
from datetime import datetime
import math
import os
import re
import sys

import jinja2

# pattern is not yet compatible with Python 3
if sys.version_info < (3, 0):
    import pattern.en

FILE = os.path.dirname(__file__)
STOPWORDS = set([w.strip() for w in open(os.path.join(FILE,
    'stopwords')).readlines()])

class TempoTFIDF(object):
    """ Python object for calculating and visualizing term weight-inverse
    document weight scores for a collection of documents produced over time,
    i.e., temporal TF-IDF scores.

    Parameters
    ----------
    stopwords : str (default=None)
        File path to custom stopwords document, with each stopword on its own
        line seperated by a newline \\n. If left blank, will use a default set
        of stopwords contained in the stopwords file.

    max_font_size : int (default=100)
        Maximum font size for visualization.

    word_regexp : string (default=None)
         Regular expression defining a word for tokenization purposes.

    collocations : bool (default=False)
        Whether to score bigrams instead of unigrams.
    """

    def __init__(self, stopwords=None, max_font_size=100, word_regexp=None,
                 collocations=False):
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.max_font_size = max_font_size
        self.word_regexp = word_regexp if word_regexp is not None else r"\w[\w']+"
        self.collocations = collocations

    def score_documents(self, documents, dates, date_format='%Y-%m-%d',
                        time_unit='month'):
        """ Main function. Cleans and scores documents according to a
        user-provided time unit.

        Parameters
        ----------
        documents : list
            Documents to be analyzed. len(documents) must eqal len(dates).

        dates : list
            Dates associated with each document. len(dates) must equal
            len(documents).

        date_format : str (default='%Y-%m-%d')
             Format of dates in dates parameter.

        time_unit : str (default='month')
            The time unit to compare term weight over. Accepts day, month, or
            year.

        Returns
        -------

        Dict corresponding to input documents and their dates

            {'February 2015': {'taste': 5.980538, 'better': 3.3423432},
             'March 2015': {'wrong': 4.343234, 'two': 1.34234234}}
        """

        # Aggregate documents according to time_unit
        aggr_dates = [self.extract_date(d, time_unit) for d in dates]
        aggr_docs = {}
        for i, doc in enumerate(documents):
            dt = aggr_dates[i]
            if dt not in aggr_docs:
                aggr_docs[dt] = []
            aggr_docs[dt].append(doc)
        time_docs = {k: ' '.join(v) for k,v in aggr_docs.items()}

        doc_tokens = {k: self.process_text(v) for k,v in time_docs.items()}
        doc_freqs = {k: self.calculate_word_frequencies(v) for k,v in doc_tokens.items()}

        # Calculate n_i- number of documents a term appears in
        n_i = {}
        for k, v in doc_freqs.items():
            for word, freq in v.items():
                if word not in n_i:
                    n_i[word] = 1
                else:
                    n_i[word] += 1
        return self.generate_from_frequencies(doc_freqs, n_i, aggr_dates)

    def process_text(self, document):
        """ Preprocesses a document into a clean list of tokens. Removes
        stopwords, punctuation, numbers, URLs, Twitter handles, etc.
        Attempts to normalize possessives and if running Python 2.x will
        attempt to singularize words.

        If self.collocations is set to True, will return bigrams.

        Parameters
        ----------
        document : str
            Documents to be cleaned

        Returns
        -------
        List of tokens
        """
        word_def = self.word_regexp
        url_def = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})"
        twitter_def = r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"
        phone_def = r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
        email_def = r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"

        document = re.sub(url_def, '', document)
        document = re.sub(twitter_def, '', document)
        document = re.sub(phone_def, '', document)
        document = re.sub(email_def, '', document)

        stopwords = set([w.lower() for w in STOPWORDS])

        words = re.findall(word_def, document, 0)
        words = [w.lower() for w in words if w.lower() not in stopwords]
        words = [w[:2] if w.endswith("'s") else w for w in words]
        words = [w for w in words if not w.isdigit()]
        if sys.version_info < (3, 0):
            words = [pattern.en.singularize(w) for w in words]
        if self.collocations:
            words = zip(words, words[1:])
        return words

    def calculate_word_frequencies(self, tokens):
        """ Counts occurances of tokens in list of tokens.

        Parameters
        ----------
        tokens : list
            List of strings (tokens) to tabulate

        Returns
        -------
        Dictionary of tokens with corresponding counts
        """
        return dict(Counter(tokens))

    def generate_from_frequencies(self, document_frequencies,
                                  collection_frequencies, dates,
                                  date_format='%Y-%m-%d', date_unit='month'):
        """ Generates the temporal TF-IDF scores for each document-term dyad
        in the collection of docoument/collection frequencies.

        Frequency of word w in time t:
            f_w_t = count(w_t)
        Inverse document frequency of word w:
            if_w = 1 + log( N / (count(w) + 1))
        Score of w:
            f_w_t * if_w

        Parameters
        ----------
        document_frequencies : dict
            A dict containing a dict for each time unit in the collection of
            documents. Each element of document_frequencies is a dict
            containing the term frequencies associated with that document. For
            example:

                document_frequencies: {'March 2001': {'compete': 4,
                                                      'travel': 1},
                                       'April 2001': {'assume': 2',
                                                      'deliver': 1}}

        collection_frequencies : dict
            A dict containing the number of documents a term appears in.

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
        def max_dict_value(d):
            """ Returns the max value v in a dict = {k: v}."""
            return max([v for k,v in d.items()])

        documents_scores = {}

        N = len(document_frequencies.keys())
        for k,v in document_frequencies.items():
            document_scores = dict()
            for token in v:
                f_w_t = v[token]
                if_w = 1 + \
                    math.log( N / (float(collection_frequencies[token]) + 1))
                s = math.pow(f_w_t, 0.5) * if_w
                document_scores[token] = s

            # Normalize scores to [0, 1]
            score_max = max_dict_value(document_scores)
            document_scores = {k: v / float(score_max) for k,v in \
                               document_scores.items()}
            documents_scores[k] = document_scores

        return documents_scores

    def visualize(self, document_scores, path='visualize.html'):
        """ Produces an HTML file to visualize change over time.

        Parameters
        ----------
        document_scores : dict
            Output of self.score_documents, the TF-IDF scores for each term.

        Returns
        -------
        Saves an HTML file to path
        """

        def render(template_path, context):
            """ Handles Jinja2 templating, loading an HTML template file
            and inserting the context variables."""
            path, filename = os.path.split(template_path)
            return jinja2.Environment(loader=jinja2.FileSystemLoader(path or
                './')).get_template(filename).render(context)

        score_font_sizes = self.generate_font_sizes(document_scores)

        context = {'score_font_sizes': score_font_sizes}
        result = render('template.html', context)

        with open(path, 'w+') as f:
            f.write(result)

        print('Visualization rendered at %s\n' % path)

    def generate_font_sizes(self, document_scores):
        """ Used for visualization to make each term's font size proportional
        to its score. Font size is calculated via:

            fontsize_w = last token's font size * (score/last token's score)

        Initial token font size is determined by max_font_size parameter.

        Parameters
        ----------
        document_scores : dict
            Output of self.score_documents, the TF-IDF scores for each term.

        Returns
        -------
        A dict containing a dict for each time unit in the collection of
            documents. Each element of document_frequencies is a dict
            containing the fontsize associated with that document-term.
        """

        def sort_dict_value(d, descending=False):
            """ Sorts a dict by magnitude of its value."""
            return sorted(d.items(), key=lambda x: x[1], reverse=descending)

        documents_font_sizes = {}
        for date, scores in document_scores.items():
            last_freq = 1.
            max_font_size = self.max_font_size
            font_size = max_font_size
            document_font_sizes = []

            scores = sort_dict_value(scores, descending=True)

            for word, score in scores:
                if font_size < 1:
                    font_size = 1
                font_size = int(font_size * ( score / float(last_freq) )) - 1
                last_freq = score
                document_font_sizes.append((word, font_size))

            documents_font_sizes[date] = document_font_sizes
        return documents_font_sizes

    @staticmethod
    def extract_date(d, part, date_format='%Y-%m-%d'):
        """ Extracts a date part from a date as a string. Can select week, month, or year.

        Parameters
        ----------
        d : string
           Date from which date part is extracted

        part : string
            Date to extract from d: week, month, or year

        date_format : string
            Format of date string d

        Returns
        -------
        A string with the requested date part
        """
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