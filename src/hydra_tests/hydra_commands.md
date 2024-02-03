
```sh
python hydra_tests/__main__.py 'pipeline/transformers@pipeline.text.transformers=[scaler]'
```

or:

```sh
python hydra_tests/__main__.py '+pipeline/transformers@pipeline.text.transformers=[scaler]'
```
