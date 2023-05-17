"""
Unit tests for xlmroberta_weighted_tfidf.py.
"""

import unittest
import math

from src.pipeline_transformers.xlmroberta_weighted_tfidf import DocumentEmbeddings, TFIDF


class TestTFIDF(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = [
            ["this", "is", "a", "test", "string", "."],
            ["a", "test", "string", "is", "used", "to", "test", "my", "code"],
            ["a", "test", "is", "testing", "testing", "testing"]
        ]
        print(cls.test_data)
        cls.tfidf = TFIDF(cls.test_data)

    # Start testing tf
    def test_N(self) -> None:
        """
            Testing N is number of documents
        """
        actual = self.tfidf.N
        exp = 3
        self.assertEqual(actual, exp)


    def test_doc3_testing_tf(self) -> None:
        """
            Testing tf greater than 2
        """
        actual = self.tfidf.get_tf((2, "testing"))
        exp = 3
        self.assertEqual(actual, exp)


    def test_doc2_test_tf(self) -> None:
        """
            Testing tf greater than 1
        """
        actual = self.tfidf.get_tf((1, "test"))
        exp = 2
        self.assertEqual(actual, exp)

    def test_doc2_unk_tf(self) -> None:
        """
            Testing tf <unk> symbol
        """
        actual = self.tfidf.get_tf((1, "asdf"))
        exp = 1
        self.assertEqual(actual, exp)

    def test_doc1_tf(self) -> None:
        """
            Testing a document with all unique tokens tf
        """
        exp = 1

        for token in self.test_data[0]:
            actual = self.tfidf.get_tf((0, token))
            self.assertEqual(exp, actual)


    # start testing idf
    def test_a_idf(self) -> None:
        """
            Testing term shows up in all documents idf
        """
        actual = self.tfidf.get_idf("a")
        exp = -0.2876820724517809
        self.assertAlmostEqual(actual, exp)


    def test_doc2_test_idf(self) -> None:
        """
            Testing term only shows up in one document idf
        """
        actual = self.tfidf.get_idf("code")
        exp = 0.4054651081081644
        self.assertAlmostEqual(actual, exp)


    def test_doc1_idf(self) -> None:
        """
            Testing term shows up in two documents idf
        """
        actual = self.tfidf.get_idf("string")
        exp = 0.0
        self.assertAlmostEqual(actual, exp)


    def test_doc2_unk_idf(self) -> None:
        """
            Testing <unk> symbol idf
        """
        actual = self.tfidf.get_idf("asdf")
        exp = 1.0986122886681098
        self.assertAlmostEqual(actual, exp)


class TestDocumentEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_something(self) -> None:
        pass

