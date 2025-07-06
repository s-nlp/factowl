# FactOwl: Blazingly fast and modern factchecker

See more on [https://s-nlp.github.io/factowl/](https://s-nlp.github.io/factowl/)

## Installation

### For development:

```bash
conda update --file longtailfacts_env.yml
conda activate ltf
pip install -e ./factowl/
python -m spacy download en_core_web_sm
python factowl/factowl/download_data.py
```

### For usage:

```
pip install factowl
```


## Usage

See `example.ipynb` for usage example.
