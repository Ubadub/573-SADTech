# Source Code

## data_prep.py
- run from root directory `python src/data_prep.py`
- data directory structure:
  ```
    data/
      mal/
        text/
          MAL_MSA_01.txt
          ...
          MAL_MSA_50.txt
        mal_train_label.tsv
      tam/
        text/
          TAM_MSA_01.txt
          ...
          TAM_MSA_59.txt
        tam_train_label.tsv
  ```
- edited tam_train_label.tsv to only contain labels for files that we have
  - TAM_MSA_ 1-24, 26-36, 38-41 (should be 39 labels total)
- file format
  ```python
    # Save sparse matrix to file
    np.savez("file.npz", data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
  ```