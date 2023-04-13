# Source Code

## `data_prep.py`
- run from root directory `python src/data_prep.py`
- outputs `data/master_text_with_labels.csv` containing file name, language code,
  label, and cleaned text for both mal and tam text data
- outputs train and val vectors for both mal and tam text data as sparse matrices
  in `outputs/vectors/text`
  - sparse matrix file format
    ```python
      # Save sparse matrix to file
      np.savez("file.npz", data=array.data, indices=array.indices,
              indptr=array.indptr, shape=array.shape)
    ```
