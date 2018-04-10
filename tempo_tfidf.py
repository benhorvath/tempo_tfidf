#!/usr/bin/env python

""" Class definition for Temporal TF-IDF object. Provides term-weighting a la
TF-IDF but for a collection of documents produced over time, i.e., associated
with some date.

TODO
----
- Add tooltip for each word with score?
- Add CLI to bottom of this script
- handle text encodings better
- Add min tokens parameter, default = 3
- Ensure order of dict dat"poaes: April 2001, January 2002, etc.
- Make original algorithm an option
- Allow larger collection of documents option, greater than sum of documents
  we're interested in
- Test on other kinds of documents
- Rewrite documentation -- consider refactoring

"""

from collections import Counter
from datetime import datetime
import math
import os
import re

import jinja2
import pattern.en

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

        # ORIGINAL
        # Calculate document collection frequencies
        # collection_tokens = []
        # for k,v in doc_tokens.items():
        #     for i in v:
        #         collection_tokens.append(i)
        # collection_freqs = self.calculate_word_frequencies(collection_tokens)

        # Calculate n_d - number of documents a term appears in
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
        stopwords, punctuation, numbers, and attempts to normalizes plurals
        and other word endings.

        If self.collocations is set to True, will return bigrams.

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
        def max_dict_value(d):
            return max([v for k,v in d.items()])

        documents_scores = {}

        # ORIGINAL
        # for k,v in document_frequencies.items():
        #     document_scores = dict()
        #     for token in v:
        #         f_w_t = v[token]
        #         if_w = math.log ( 1 / float(collection_frequencies[token]) )
        #         s = f_w_t * if_w
        #         document_scores[token] = s
        #     documents_scores[k] = document_scores

        N = len(document_frequencies.keys())
        for k,v in document_frequencies.items():
            document_scores = dict()
            for token in v:
                f_w_t = v[token]
                if_w = 1 + math.log( N / (float(collection_frequencies[token]) + 1))
                s = math.pow(f_w_t, 0.5) * if_w
                ## ORIGINAL
                #if_w = math.log ( 1 / float(collection_frequencies[token]) )
                #s = f_w_t * if_w

                document_scores[token] = s

            # Normalize scores to [0, 1]
            score_max = max_dict_value(document_scores)
            document_scores = {k: v / float(score_max) for k,v in document_scores.items()}

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

        print('\nVisualization rendered at %s\n' % path)

    def generate_font_sizes(self, document_scores):
        """ Exact:

        fontsize_{w_{i}} =  fontsize_{w_{i-1}} \times \frac{s\cdot core_{w_{i}}}{score_{w_{i-1}}} - 1

        which is approximately fontsize^* = fontsize_{t-1} * delta frequency
        """

        def sort_dict_value(d, descending=False):
            return sorted(d.items(), key=lambda x: x[1], reverse=descending)

        documents_font_sizes = {}
        for date, scores in document_scores.items():
            last_freq = 1.
            max_font_size = 100
            font_size = max_font_size
            document_font_sizes = []

            scores = sort_dict_value(scores, descending=True)

            for word, score in scores:
                rs = .5
                if font_size < 1:
                    font_size = 1
                font_size = int(font_size * ( score / float(last_freq) )) - 1
                last_freq = score
                document_font_sizes.append((word, font_size))

            documents_font_sizes[date] = document_font_sizes
        return documents_font_sizes

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


    # docs = ['While troubleshooting HIVE performance issues when TEZ engine is being used there may be a need to increase the number of mappers used during a query.', 'In Monday\'s damp predawn darkness, teachers gathered in front of Muskogee High. But instead of heading to their classrooms, they piled on to a bus painted with the schoo\'s mascot the Roughers and headed 150 miles west to Oklahoma City.', "Tip 1: Partitioning Hive Tables Hive is a powerful tool to perform queries on large data sets and it is particularly good at queries that require full table scans. Yet many queries run on Hive have filtering where clauses limiting the data to be retrieved and processed, e.g. SELECT * WHERE . Hive users tend to have or develop a domain knowledge, understand the data they work with and the queries commonly executed or scheduled. With this knowledge we can identify common data structures that surface in queries. This enables us to identify columns with a (relatively) low cardinality like geographies or dates and high relevance to key queries. For example, common approaches to slice the airline data may be by origin state for reporting purposes. We can utilize this knowledge to organise our data by this information and tell Hive about it. Hive can utilize this knowledge to exclude data from queries before even reading it. Hive tables are linked to directories on HDFS or S3 with files in them interpreted by the meta data stored with Hive. Without partitioning Hive reads all the data in the directory and applies the query filters on it. This is slow and expensive since all data has to be read. In our example a common reports and queries might be generated on an origin state basis. This enables us to define at creation time of the table the state column to be a partition. Consequently, when we write data to the table the data will be written in sub-directories named by state (abbreviations). Subsequently, queries filtering by origin state, e.g. allow Hive to skip all but the relevant sub-directories and data files. This can lead to tremendous reduction in data required to read and "]


    # dts = ['2018-01-01', '2018-01-02', '2018-02-01']

    # scorer = TempoTFIDF()
    # doc_scores = scorer.score_documents(docs, dts,time_unit='week')
    # print(doc_scores)
    # scorer.visualize(doc_scores)

    import pandas as pd

    df = pd.read_csv('ken_lay_emails.csv')
    docs = df['message'].tolist()
    dates = df['date'].tolist()

    scorer = TempoTFIDF()

    doc_scores = scorer.score_documents(docs, dates, time_unit='week')
    font_sizes = scorer.generate_font_sizes(doc_scores)
    scorer.visualize(doc_scores)
