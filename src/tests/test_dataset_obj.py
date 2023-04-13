"""
Unit tests for dataset_obj.py.
"""

import unittest

from src.dataset_obj import get_data, create_dataset_obj

class TestGetData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mal_df = get_data("data", "mal")
        cls.tam_df = get_data("data", "tam")


    def test_mal_data_length(self):
        actual = len(self.mal_df)
        exp = 70
        self.assertEqual(actual, exp)


    def test_tam_data_length(self):
        actual = len(self.tam_df)
        exp = 64
        self.assertEqual(actual, exp)



    def test_MAL_MSA_01_label(self):
        # tests MAL_MSA_01 has correct label
        actual = self.mal_df[self.mal_df["file_name"] == "MAL_MSA_01"].iloc[-1]["label"]
        exp = "POSITIVE"
        self.assertEqual(actual, exp)


    def test_MAL_MSA_01_text(self):
        # tests MAL_MSA_01 contains correct text
        actual = self.mal_df[self.mal_df["file_name"] == "MAL_MSA_01"].iloc[-1]["text"]
        exp = "അവസാനം ഈ സിനിമയിലെ ഡയറി വാങ്ങി വായിക്കുന്ന 10 മിനിറ്റ്, അതാണ് ഈ സിനിമയുടെ ട്വിസ്റ്റ്‌ ആൻഡ് ടേൺ എന്ന് പറയുന്നത്;"
        self.assertTrue(exp in actual)


    def test_MAL_MSA_55_label(self):
        # tests MAL_MSA_55 has correct label
        actual = self.mal_df[self.mal_df["file_name"] == "MAL_MSA_55"].iloc[-1]["label"]
        exp = "HIGHLY NEGATIVE"
        self.assertEqual(actual, exp)


    def test_MAL_MSA_55_text(self):
        # tests MAL_MSA_01 contains correct text
        actual = self.mal_df[self.mal_df["file_name"] == "MAL_MSA_55"].iloc[-1]["text"]
        exp = "ിപ്പെടുത്താത്ത സിനിമ. എന്നിലെ പ്രേക്ഷകനെ ഒരുതരത്തിലും, പാട്ട് ആയാലും, ഫൈറ്റ് ആയാലും, എന്ത് "
        self.assertTrue(exp in actual)

    def test_TAM_MSA_01_label(self):
        # tests MAL_MSA_01 has correct label
        actual = self.tam_df[self.tam_df["file_name"] == "TAM_MSA_01"].iloc[-1]["label"]
        exp = "POSITIVE"
        self.assertEqual(actual, exp)


    def test_TAM_MSA_01_text(self):
        # tests MAL_MSA_01 contains correct text
        actual = self.tam_df[self.tam_df["file_name"] == "TAM_MSA_01"].iloc[-1]["text"]
        exp = "ி.ஜி.எம் அப்படின்னு வரப்போ , ரொம்ப எல்லாம் அப்படியே மண்டையெல்லாம் குழப்பி எல்லாம் மியூசிக் பண்ணல ."
        self.assertTrue(exp in actual)


    def test_TAM_MSA_55_label(self):
        # tests MAL_MSA_55 has correct label
        actual = self.tam_df[self.tam_df["file_name"] == "TAM_MSA_55"].iloc[-1]["label"]
        exp = "HIGHLY POSITIVE"
        self.assertEqual(actual, exp)


    def test_TAM_MSA_55_text(self):
        # tests MAL_MSA_01 contains correct text
        actual = self.tam_df[self.tam_df["file_name"] == "TAM_MSA_55"].iloc[-1]["text"]
        exp = "எழுதின ஒரு பாட்டு. நம்மோடு, நம்மோடு, நம்மோடு ன்னு, அது வீட்டுக்கு வர வரைக்கும் நம்மோடு "
        self.assertTrue(exp in actual)


class TestCreateDataSetObject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mal_train, cls.mal_dev = create_dataset_obj(dir_path="data", lang="mal")
        cls.tam_train, cls.tam_dev = create_dataset_obj(dir_path="data", lang="tam")


    def test_MAL_MSA_01_label(self):
        data = self.mal_train if "MAL_MSA_01" in self.mal_train["file_name"] else self.mal_dev
        index = data['file_name'].index("MAL_MSA_01")
        actual = data[index]["label"]
        exp = 3
        self.assertEqual(actual, exp)


    def test_MAL_MSA_01_text(self):
        data = self.mal_train if "MAL_MSA_01" in self.mal_train["file_name"] else self.mal_dev
        index = data['file_name'].index("MAL_MSA_01")
        actual = data[index]["text"]
        exp = "അവസാനം ഈ സിനിമയിലെ ഡയറി വാങ്ങി വായിക്കുന്ന 10 മിനിറ്റ്, അതാണ് ഈ സിനിമയുടെ ട്വിസ്റ്റ്‌ ആൻഡ് ടേൺ എന്ന് പറയുന്നത്;"
        self.assertTrue(exp in actual)


    def test_MAL_MSA_23_label(self):
        data = self.mal_train if "MAL_MSA_23" in self.mal_train["file_name"] else self.mal_dev
        index = data['file_name'].index("MAL_MSA_23")
        actual = data[index]["label"]
        exp = 3
        self.assertEqual(actual, exp)


    def test_MAL_MSA_23_text(self):
        data = self.mal_train if "MAL_MSA_23" in self.mal_train["file_name"] else self.mal_dev
        index = data['file_name'].index("MAL_MSA_23")
        actual = data[index]["text"]
        exp = "ഇപ്പോൾ മലയാളത്തിൽ ഇറങ്ങിയിട്ടുള്ള, ആൻ മരിയ കലിപ്പിലാണ് , അങ്ങനെയുള്ള, അതുപോലെതന്നെ മങ്കിപെൻ പോലെയുള്ള"
        self.assertTrue(exp in actual)


    def test_TAM_MSA_01_label(self):
        data = self.tam_train if "TAM_MSA_01" in self.tam_train["file_name"] else self.tam_dev
        index = data['file_name'].index("TAM_MSA_01")
        actual = data[index]["label"]
        exp = 3
        self.assertEqual(actual, exp)


    def test_TAM_MSA_01_text(self):
        data = self.tam_train if "TAM_MSA_01" in self.tam_train["file_name"] else self.tam_dev
        index = data['file_name'].index("TAM_MSA_01")
        actual = data[index]["text"]
        exp = "ி.ஜி.எம் அப்படின்னு வரப்போ , ரொம்ப எல்லாம் அப்படியே மண்டையெல்லாம் குழப்பி எல்லாம் மியூசிக் பண்ணல ."
        self.assertTrue(exp in actual)


    def test_TAM_MSA_23_label(self):
        data = self.tam_train if "TAM_MSA_23" in self.tam_train["file_name"] else self.tam_dev
        index = data['file_name'].index("TAM_MSA_23")
        actual = data[index]["label"]
        exp = 2
        self.assertEqual(actual, exp)


    def test_TAM_MSA_23_text(self):
        data = self.tam_train if "TAM_MSA_23" in self.tam_train["file_name"] else self.tam_dev
        index = data['file_name'].index("TAM_MSA_23")
        actual = data[index]["text"]
        exp = "இவங்க வந்து, இந்த டயலாக் காமெடி மட்டுமே நம்பி, இந்தப் படம் வந்து எடுக்கப்பட்டு இருக்கு . இதுல"
        self.assertTrue(exp in actual)


if __name__ == '__main__':
    unittest.main()