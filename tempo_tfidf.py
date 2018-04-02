#!/usr/bin/env python

""" Class definition for Temporal TF-IDF object. Provides term-weighting a la
TF-IDF but for a collection of documents produced over time, i.e., associated
with some date.

TODO
----
- Add functions to collocation=True feature.
- score_documents -- add function to convert dates to date_unit before
  calculating frequencies, something like:
      # documents contains the list of documents
      original_dates = set(dates)
      transformed_dates = [get_date(x, 'month') for x in dates]
      transformed_tokens = dict{}
      for x, i in documents.index():
          transformed = transformed_dates[i]
          transformed_tokens[i].append(x)
      final_text = {k: ' '.join(v) for k,v in transformed_tokens.iteritems()}


"""

from collections import Counter
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
        """ Clean document text and produce term weights.

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

            {'document1':
                {'taste': 5.9808,
                 'better': 3.34},
             'document2':
                {'wrong': 4.343,
                 'two': 1.34}}
        """
        if time_unit = 'day':
            print('day')
        elif time_unit = 'month':
            print('month')
        elif time_unit = 'year':
            print('year')
        else:
            print('Please select a valid time unit: day, month, year')
            break

        doc_tokens = [self.process_text(doc) for doc in self.documents]
        doc_freqs = [self.calculate_word_frequencies(t) for t in tokens]

        collection_tokens = [token for doc in doc_tokens for token in doc]
        collection_freqs = self.calculate_word_frequencies(collection_tokens)
        return self.generate_from_frequencies(doc_freqs, collection_freqs)

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
        List of dicts, each dict corresponding to an input document

            [{'taste': 5.980538, 'better': 3.3423432},  # document 1
             {'wrong': 4.343234, 'two': 1.34234234}}    # document 2
        """
        documents_scores = []

        for document in document_frequencies:
            document_scores = dict()
            for k in document.keys():
                f_w_t = document[k]
                if_w = math.log( 1 / float(collection_frequencies[k]))
                s = abs(f_w_t * if_w)
                document_scores[k] = s
            documents_scores.append(document_scores)

        return documents_scores


if __name__ == '__main__':

    docs = ['While troubleshooting HIVE performance issues when TEZ engine is being used there may be a need to increase the number of mappers used during a query.', 'In Monday\'s damp predawn darkness, teachers gathered in front of Muskogee High. But instead of heading to their classrooms, they piled on to a bus painted with the schoo\'s mascot the Roughers and headed 150 miles west to Oklahoma City.', 'In this query, you can create dimensions from customer_id and lifetime_spend. However, suppose you wanted the user to be able to specify the region, instead of hard-coding it to "northeast". The region cannot be exposed as a dimension, and therefore the user cannot filter on it as normal.']

    dts = ['2018-01-01', '2018-02-01', '2018-03-01']

    #scorer = TempoTFIDF(documents=docs, dates=dts, time_unit='month')

    scorer = TempoTFIDF()

    tokens = [scorer.process_text(doc) for doc in docs]

    docs_freq = [scorer.calculate_word_frequencies(doc) for doc in tokens]

    all_docs = ' '.join(docs)
    all_docs_tokens = scorer.process_text(all_docs)
    all_docs_freq = scorer.calculate_word_frequencies(all_docs_tokens)

    #print(docs_freq)
    #print(all_docs_freq)

    scores = scorer.generate_from_frequencies(docs_freq, all_docs_freq, dts)
    print(scores)
