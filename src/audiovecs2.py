from datasets import load_from_disk
from datasets.features import Audio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL = "facebook/wav2vec2-large-960h"
NUM_SAMPLES = 10

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tam_ds = load_from_disk("../data/tam/train_dataset_dict_audio/")["train"]

a16k = Audio(sampling_rate=16000)

with torch.no_grad():
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2Model.from_pretrained(MODEL)
    model.to(device)
    X = [a16k.decode_example(x)["array"] for x in tam_ds.to_pandas()["audio"]]
    inputs = processor(
        # X, sampling_rate=16000, return_tensors="pt", padding="longest"
        X[:NUM_SAMPLES], sampling_rate=16000, return_tensors="pt", padding="longest"
    ).to(device)
    outputs = model(**inputs, output_hidden_states=True)

    print(outputs)
    print("len(outputs.hidden_states):", len(outputs.hidden_states))

    # print("outputs.hidden_states[0].is_cuda:", outputs.hidden_states[0].is_cuda)
    # print("outputs.hidden_states[0].shape:", outputs.hidden_states[0].shape)
    # print(
    #     "outputs.hidden_states[0].element_size():",
    #     outputs.hidden_states[0].element_size(),
    # )
    # print("outputs.hidden_states[0].nelement():", outputs.hidden_states[0].nelement())
    # print(
    #     "outputs.hidden_states[0] total size:",
    #     outputs.hidden_states[0].nelement() * outputs.hidden_states[0].element_size(),
    # )

    print("outputs.hidden_states[1].is_cuda:", outputs.hidden_states[1].is_cuda)
    print("outputs.hidden_states[1].shape:", outputs.hidden_states[1].shape)
    print(
        "outputs.hidden_states[1].element_size():",
        outputs.hidden_states[1].element_size(),
    )
    print("outputs.hidden_states[1].nelement():", outputs.hidden_states[1].nelement())
    print(
        "outputs.hidden_states[1] total size:",
        outputs.hidden_states[1].nelement() * outputs.hidden_states[1].element_size(),
    )
