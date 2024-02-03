import sys
from typing import Literal

from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import (
    AlbertTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBartForConditionalGeneration,
)

BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"

# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']
LANG_IDS = {"mal": "<2ml>", "tam": "<2ta>"}


def generate(
    input_text: str,
    lang: Literal["mal", "tam"],
    model: str = "ai4bharat/MultiIndicParaphraseGenerationSS",
    max_length_perc: float = 0.1,
    min_length_perc: float = 0.1,
    device="cuda:0",
    **generation_kwargs,
):
    lang_id = LANG_IDS[lang]
    tokenizer = AutoTokenizer.from_pretrained(
        model, do_lower_case=False, use_fast=False, keep_accents=True
    )
    # tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS", do_lower_case=False, use_fast=False, keep_accents=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
    # model = MBartForConditionalGeneration.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS")

    # Some initial mapping
    bos_id = tokenizer._convert_token_to_id_with_added_voc(BOS)
    eos_id = tokenizer._convert_token_to_id_with_added_voc(EOS)
    pad_id = tokenizer._convert_token_to_id_with_added_voc(PAD)

    # First tokenize the input. The format below is how IndicBART was trained so the
    # input should be "Sentence </s> <2xx>" where xx is the language code. Similarly,
    # the output should be "<2yy> Sentence </s>".

    tok_out = tokenizer(
        # tam_ds_dict["train"][0]["text"] + f" </s> {lang_id}",
        input_text + f" {EOS} {lang_id}",
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to(device)

    num_inp_tokens = len(tok_out.input_ids.squeeze())

    max_length = int(num_inp_tokens * (1 + max_length_perc))
    min_length = int(num_inp_tokens * (1 - min_length_perc))

    print("num_inp_tokens:", num_inp_tokens)
    print("max_length:", max_length)
    print("min_length:", min_length)

    # For generation. Pardon the messiness. Note the decoder_start_token_id.

    model_output = model.generate(
        tok_out.input_ids,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        max_length=max_length,
        min_length=min_length,
        decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc(lang_id),
        **generation_kwargs,
    )

    # print(model_output)

    # Decode to get output strings
    decoded_output = tokenizer.decode(
        model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded_output




@hydra.main(version_base=None, config_path="../config/augmentation", config_name="config")
def main(cfg: DictConfig) -> None:
# def main():
    print(OmegaConf.to_yaml(cfg, resolve=False))
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    lang = sys.argv[1]
    # ds_dict = load_from_disk(f"../data/{lang}/full_dataset_dict")
    ds_dict = load_from_disk(cfg.dataset_path)
    dses = [ds_dict[_] for _ in cfg.splits_to_augment]

    for ds in dses:
        print(
            generate(
                input_text=ds["text"][int(sys.argv[2])],
                lang=lang,
                early_stopping=True,
                encoder_no_repeat_ngram_size=6,
                # encoder_repetition_penalty=0.1,
                no_repeat_ngram_size=6,
                num_beams=1,
                penalty_alpha=0.6,
                top_k=4,
                use_cache=True,
                output_scores=True,
                do_sample=False,
            )
        )

        # print(
        #     generate(
        #         input_text=ds["text"][int(sys.argv[2])],
        #         lang=lang,
        #         early_stopping=True,
        #         encoder_no_repeat_ngram_size=3,
        #         # encoder_repetition_penalty=0.1,
        #         no_repeat_ngram_size=3,
        #         num_beams=5,
        #         top_k=1000,
        #         use_cache=True,
        #         output_scores=True,
        #         do_sample=True,
        #     )
        # )


if __name__ == "__main__":
    main()
