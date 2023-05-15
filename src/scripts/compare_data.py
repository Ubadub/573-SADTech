"""
"""

import os
import pandas as pd


def get_df(directory):
    files = os.listdir(directory)
    data_dict = {"file_name": [], "text_data": []}
    for file in files:
        fname = file.split(".")[0]
        fpath = os.path.join(directory, file)
        with open(fpath, encoding="utf-8-sig") as f:
            lines = f.readlines()
            text = "".join(l.strip() for l in lines)
            data_dict["file_name"].append(fname)
            data_dict["text_data"].append(text)
    df = pd.DataFrame(data_dict)
    return df


def get_merged_df(lang):
    real_labels = pd.read_csv(f"../data/{lang}/all.csv")
    real_labels = real_labels.rename(columns={"file": "file_name", "label": "label_real"})

    real_df = get_df(f"../data/{lang}/text")
    real_df = real_df.merge(real_labels, on="file_name")

    test_df = get_df(f"../data/test_data/{lang}/text")
    # test_df = pd.DataFrame({"file_name": ["MAL_MSA_04"], "text_data": ["അലക്സ് അവിടെ ചെന്ന് കഴ്ഞ്ഞിട്ട്, കാര്യങ്ങളും അത് ഞാൻ ഇപ്പോൾ റിവീൽ ചെയ്യുന്നില്ല , പലകാര്യങ്ങളും അലെക്സിനെയും ആ ലേഡിയെയും ചുരുളുകൾ അഴിയുന്നു പോലെഎല്ലാം ട്വിസ്റ്റ് ആഹ് , അവിടം മുതൽ ആദ്യത്തെ ഒരു മുപ്പത് മിനിറ്റു ഇപ്പോൾ ഞാൻ പറഞ്ഞ കാര്യങ്ങൾ ഒക്കെ ആദ്യത്തെ മുപ്പതു മിനിറ്റ് ൽ ഉള്ളതാ . അത് കഴിഞ്ഞ്‌ ആ മോർച്ചറിയിൽ ചെന്ന് കഴിഞ്ഞിട്ട് , പിന്നെ നടക്കുന്നത് മൊത്തം ട്വിസ്റ്റാ , ഫുൾ ട്വിസ്റ്റാ, ഇതിന്റെ ബാക്കി എന്താണെന്നു , എന്നുള്ളതാണ് പടത്തിന്റെ മൊത്തം കഥ . നിങ്ങൾ ആരും പ്രതീക്ഷിക്കാത്ത, ആരും വിചാരിക്കാത്ത ഒരു ട്വിസ്റ്റ് ആൻഡ് സസ്പെൻസ് ആണ് ഇതിന്റെ ക്ലൈമാക്സ്. നിങ്ങൾ എന്തായാലും ഉറപ്പായും വാച്ച് ചെയ്യേണ്ട ഒരു മൂവി ആണ്. ഈ ട്വിസ്റ്റ്കളിഷ്ട്ടപെടുന്ന, അല്ലെങ്കിൽ സസ്പെന്സ്ത്രില്ലറുകൾ ഇഷ്ട്ടപെടുന്ന ആൾക്കാർ ഉണ്ടെങ്കിൽ അവർക്കുവേണ്ടിയുള്ള ഒരു സിനിമ യാണ് ഇത് . ഇത് സ്പാനിഷ് ആണ്, സബ്‌ടൈറ്റിൽ ഒക്കെ നമുക്ക് ഡൗലോഡ് ചെയ്യാം, നിങ്ങൾക്ക് എം.എക്സ് പ്ലയെർ ഒക്കെ സബ്‌ടൈറ്റിൽ സെർച്ച് ചെയ്തു ഡൌൺലോഡ് ചെയ്യാൻ പറ്റും. ഈസി ആയി ഡൌൺലോഡ് ചെയ്യാം. സൊ, ലാംഗ്വേജ്ന്റെ പ്രെശ്നം വരുന്നില്ല. എന്തായാലും ഈപടം ധൈര്യമായി വാച്ച് ചെയ്യാം സൂപ്പർ പടമാണ്. ഒരു അന്യായ ട്വിസ്റ്റും ക്ലൈമാക്സും ഒക്കെയാണ് . ഈ പടത്തിന് വുഡ്‌സ് ഡേയ്സ് സിനിമാസ് കൊടുക്കുന്ന റേറ്റിംഗ് സെവൻ പോയിന്റ് ഫൈവ് ഔട്ട് ഓഫ് ടെൻ."]})

    merged_df = real_df.merge(test_df, on="text_data", how="outer", suffixes=["_real", "_test"])
    merged_df = merged_df[["file_name_real", "label_real", "file_name_test", "text_data"]]

    return merged_df


def main():
    mal_merged_df = get_merged_df("mal")
    mal_merged_df.to_csv("../data/mal/merged.csv")

    tam_merged_df = get_merged_df("tam")
    tam_merged_df.to_csv("../data/tam/merged.csv")


if __name__ == "__main__":
    main()