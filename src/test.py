"""
Testing
Unit tests for python scripts.
"""

from utils import assert_equals

from data_prep import clean_up_line
from spacy.lang.ta import Tamil

def test_data_prep():
    print("------------------------------------------------------------")
    print("Testing functions in data_prep.py")

    print("    Testing clean_up_line")

    TAM_STOP_WORDS = Tamil().Defaults.stop_words

    print("        Test 1")
    # testing removal of punctuation at end of word
    #   not entirely sure what to do with .சோ - currently does nothing
    test = "தான் வந்து இந்த படத்தில் மிகப்பெரிய பொருளா அமைஞ்சிருக்கு .சோ, அப்படி"
    exp = "படத்தில் மிகப்பெரிய பொருளா அமைஞ்சிருக்கு .சோ அப்படி"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("        Test 2")
    # not entirely sure what to do with பார்த்தோம்.d - currently does nothing
    test = "படத்தினை பத்தின ஒரு சின்ன அவுட்லைன் வந்து நம்ம பார்த்தோம்.d இந்தப்"
    exp = "படத்தினை பத்தின சின்ன அவுட்லைன் நம்ம பார்த்தோம்.d"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("        Test 3")
    # testing removal of period at end of line
    test = "யாரும் பாக்கல என்றால் டஃபினிட் போய் பாருங்கள்."
    exp = "யாரும் பாக்கல என்றால் டஃபினிட் போய் பாருங்கள்"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("        Test 4")
    # not entirely sure what to do with ஏ1 - currently just removes the whole token
    # also see test 6
    test = "வீக்என்டுக்கு ஏ1 வந்து பயங்கரமா ஸ்கோர் பண்ணியிருக்கிறாரு அப்படின்னுதான்"
    exp = "வீக்என்டுக்கு பயங்கரமா ஸ்கோர் பண்ணியிருக்கிறாரு அப்படின்னுதான்"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("        Test 5")
    # testing removal of individual numbers
    test = "வந்து ஒரு 12 சூப்பர் ஹீரோஸ் வச்சு வந்து டைரக்ட் பண்ணியிருக்காங்க. போன"
    exp = "சூப்பர் ஹீரோஸ் வச்சு டைரக்ட் பண்ணியிருக்காங்க போன"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("        Test 6")
    # testing removal of numbers with decimal and words with numbers in them
    test = "6.75 அவுட் ஆஃப்10 ."
    exp = "அவுட்"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("        Test 7")
    # testing removal of quotes
    test = "நான் உன் போன் ஏன் போனா மிஸ்டேக் பண்ணி எடுத்துட்டு போயிட்டேன்\" . \" சோ,"
    exp = "போன் போனா மிஸ்டேக் பண்ணி எடுத்துட்டு போயிட்டேன் சோ"
    assert_equals(exp, clean_up_line(test, TAM_STOP_WORDS))

    print("    Success")
    print("------------------------------------------------------------")


def main():
    test_data_prep()


if __name__ == '__main__':
    main()