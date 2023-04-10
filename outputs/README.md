# Outputs

## vectors/
- files
  - tam_X_train_tfidf.npz - Tamil train vectors
  - tam_X_val_tfidf.npz - Tamil validation vectors
  - mal_X_train_tfidf.npz - Malayalam train vectors
  - mal_X_val_tfidf.npz - Malayalam validation vectors
- file format
  ```python
    from scipy import sparse

    # Load sparse matrix from file
    loader = np.load("file.npz")
    new_csr =  sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                  shape=loader['shape'])
  ```
