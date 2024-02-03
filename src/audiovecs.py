from datasets import load_from_disk
from datasets.features import Audio
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# NUM_SAMPLES = 10
NUM_SPLITS = 5

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tam_ds = load_from_disk("../data/tam/train_dataset_dict_audio/")["train"]
X = tam_ds.to_pandas()["audio"]

a16k = Audio(sampling_rate=16000)
X = X.map(lambda x: a16k.decode_example(x)["array"])
# X = [a16k.decode_example(x)["array"] for x in X]

with torch.no_grad():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    model.to(device)

    for batch_idx, batch in enumerate(np.array_split(X, len(X)/NUM_SPLITS)):
        batch_inputs = processor(
            batch, sampling_rate=16000, return_tensors="pt", padding="longest"
            # X, sampling_rate=16000, return_tensors="pt", padding="longest"
            # X[:NUM_SAMPLES], sampling_rate=16000, return_tensors="pt", padding="longest"
        ).to(device)
        batch_outputs = model(**batch_inputs, output_hidden_states=True)

        hidden_states = np.array([hidden.detach().cpu().numpy() for hidden in batch_outputs.hidden_states])
        out_path = f"
        hidden_states.save(

        # gc.collect()
        # torch.cuda.empty_cache()

        print(batch_outputs)
        print("len(batch_outputs.hidden_states):", len(batch_outputs.hidden_states))

        # print("batch_outputs.hidden_states[0].is_cuda:", batch_outputs.hidden_states[0].is_cuda)
        # print("batch_outputs.hidden_states[0].shape:", batch_outputs.hidden_states[0].shape)
        # print(
        #     "batch_outputs.hidden_states[0].element_size():",
        #     batch_outputs.hidden_states[0].element_size(),
        # )
        # print("batch_outputs.hidden_states[0].nelement():", batch_outputs.hidden_states[0].nelement())
        # print(
        #     "batch_outputs.hidden_states[0] total size:",
        #     batch_outputs.hidden_states[0].nelement() * batch_outputs.hidden_states[0].element_size(),
        # )

        print("batch_outputs.hidden_states[1].is_cuda:", batch_outputs.hidden_states[1].is_cuda)
        print("batch_outputs.hidden_states[1].shape:", batch_outputs.hidden_states[1].shape)
        print(
            "batch_outputs.hidden_states[1].element_size():",
            batch_outputs.hidden_states[1].element_size(),
        )
        print("batch_outputs.hidden_states[1].nelement():", batch_outputs.hidden_states[1].nelement())
        print(
            "batch_outputs.hidden_states[1] total size:",
            batch_outputs.hidden_states[1].nelement() * batch_outputs.hidden_states[1].element_size(),
        )
