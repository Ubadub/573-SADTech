"""
Unit tests for data_prep.py.
"""

import unittest

from src.data_prep import clean_up_line
from spacy.lang.ta import Tamil


class TestCleanUpLine(unittest.TestCase):
    def setUp(self):
        self.tam_stop_words = Tamil().Defaults.stop_words

    def test_stop_words(self):
        # not entirely sure what to do with பார்த்தோம்.d - currently does nothing
        actual = "படத்தினை பத்தின ஒரு சின்ன அவுட்லைன் வந்து நம்ம பார்த்தோம்.d இந்தப்"
        exp = "படத்தினை பத்தின சின்ன அவுட்லைன் நம்ம பார்த்தோம்.d"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)

    def test_end_punc(self):
        # testing removal of punctuation at end of word
        #   not entirely sure what to do with .சோ - currently does nothing
        actual = "தான் வந்து இந்த படத்தில் மிகப்பெரிய பொருளா அமைஞ்சிருக்கு .சோ, அப்படி"
        exp = "படத்தில் மிகப்பெரிய பொருளா அமைஞ்சிருக்கு .சோ அப்படி"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)

        # testing removal of period at end of line
        actual = "யாரும் பாக்கல என்றால் டஃபினிட் போய் பாருங்கள்."
        exp = "யாரும் பாக்கல என்றால் டஃபினிட் போய் பாருங்கள்"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)

    def test_num(self):
        # not entirely sure what to do with ஏ1 - currently just removes the whole token
        actual = "வீக்என்டுக்கு ஏ1 வந்து பயங்கரமா ஸ்கோர் பண்ணியிருக்கிறாரு அப்படின்னுதான்"
        exp = "வீக்என்டுக்கு பயங்கரமா ஸ்கோர் பண்ணியிருக்கிறாரு அப்படின்னுதான்"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)

        # testing removal of individual numbers
        actual = "வந்து ஒரு 12 சூப்பர் ஹீரோஸ் வச்சு வந்து டைரக்ட் பண்ணியிருக்காங்க. போன"
        exp = "சூப்பர் ஹீரோஸ் வச்சு டைரக்ட் பண்ணியிருக்காங்க போன"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)

        # testing removal of numbers with decimal and words with numbers in them
        actual = "6.75 அவுட் ஆஃப்10 ."
        exp = "அவுட்"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)

    def test_quotes(self):
        # testing removal of quotes
        actual = "நான் உன் போன் ஏன் போனா மிஸ்டேக் பண்ணி எடுத்துட்டு போயிட்டேன்\" . \" சோ,"
        exp = "போன் போனா மிஸ்டேக் பண்ணி எடுத்துட்டு போயிட்டேன் சோ"
        self.assertEqual(clean_up_line(actual, self.tam_stop_words), exp)


# class TestOther(unittest.TestCase):
#     def test_other(self):
#         self.assertEqual("a", "a")


if __name__ == '__main__':
    unittest.main()
