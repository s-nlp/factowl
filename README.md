# FactOwl: Blazingly fast and modern factchecker

See more on [https://s-nlp.github.io/factowl/](https://s-nlp.github.io/factowl/)

## Usage Examples

1. [Fact generation and verification using pre-computed contexts](https://github.com/s-nlp/factowl/blob/main/examples/eval_no_retrieve.ipynb) only on RiDiC dataset

2. [Fact generation and verification on FactScore dataset](https://github.com/s-nlp/factowl/blob/main/examples/factowl_example.ipynb) usign either fixed Wikipedia dump or Wikipedia search for context retrieval

3. LLM and rule-based [fact filtration](https://github.com/s-nlp/factowl/blob/main/examples/filter_facts.ipynb)

## Installation

### For development:

```bash
conda update --file factowl.yml
conda activate ltf
pip install -e ./factowl/
python -m spacy download en_core_web_sm
python factowl/factowl/download_data.py
```

and also download enwiki dump from [this link](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I) and place it into `./data/` directory.

### For usage:

```
pip install factowl
```


## Usage

To run the scorer on the provided sample dataset, simply use:

```
python -m factowl --input_file data/sample_input.json
```

You can also provide your own data into the script in the following format:

```
{
  "topics": [
    "{topic_id}": "{topic_name}"
  ],
  "generations": [
    "{topic_id}": "{generation}"
  ]
}
```
