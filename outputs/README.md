# Outputs

## vectors/
- files
  - `text/tam_X_train_tfidf.npz` - Tamil train vectors of text data
  - `text/tam_X_val_tfidf.npz` - Tamil validation vectors of text data
  - `text/mal_X_train_tfidf.npz` - Malayalam train vectors of text data
  - `text/mal_X_val_tfidf.npz` - Malayalam validation vectors of text data
- file format
  ```python
    from scipy import sparse

    # Load sparse matrix from file
    loader = np.load("file.npz")
    new_csr =  sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                  shape=loader['shape'])
  ```
