# Evaluation dependencies

Install the Python evaluation extras with:

```bash
pip install -r requirements-eval.txt
```

MMseqs is also required for the MMseqs-based evaluation scripts. It is a system
dependency, not a pip package; install it separately and make sure `mmseqs` is on
`PATH`, or pass the relevant script option for an explicit MMseqs executable path.
