# 🦉 FactOWL: A Cost-Efficient Tool for Long-Form Factuality Evaluation

<p align="center">
  <a href="https://github.com/s-nlp/factowl"><img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub"></a>
  <a href="https://doi.org/10.1145/3805712.3808373"><img src="https://img.shields.io/badge/SIGIR-2026_Paper-FF6B6B?logo=acm" alt="Paper"></a>
  <a href="https://pypi.org/project/factowl/"><img src="https://img.shields.io/pypi/v/factowl?color=green" alt="PyPI"></a>
  <a href="https://github.com/s-nlp/factowl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
</p>

**FactOWL** is a blazingly fast, open-source, and fully reproducible factuality evaluation tool for long-form Large Language Model (LLM) generations. 

Built to overcome the limitations of pioneering tools like FActScore and SAFE, FactOWL eliminates the need for expensive proprietary APIs and outdated static knowledge bases. By leveraging **Open LLMs (Llama-3-8B)**, **real-time Wikipedia search**, and **Wikidata triple aggregation**, FactOWL effectively resolves entity ambiguity and delivers human-aligned factuality precision at a fraction of the computational cost.

🚀 **Key Highlights:**
* ⚡ **Blazingly Fast:** Up to **10× faster** than FActScore and **20× faster** than SAFE.
* 🌐 **Real-Time & Multi-Source:** Dynamically queries live Wikipedia and Wikidata, solving the "outdated knowledge dump" problem.
* 💰 **Cost-Efficient:** 100% open-source pipeline. No paid search engines or proprietary LLM APIs required.
* 🎯 **High Alignment:** Achieves stable, human-aligned factuality precision without complex non-parametric post-processing.
* 🧩 **Modular Architecture:** Easily swap retrievers, rerankers (GTR/BM25), or judge LLMs.

---

## 🏗️ Architecture & Pipeline

FactOWL decomposes factual evaluation into three transparent, robust stages:

1. **Atomic Fact Extraction:** Prompts an open LLM to break down long-form text into self-contained, unambiguous atomic facts, explicitly resolving anaphoric pronouns and filtering out unrelated common-sense claims.
2. **Evidence Retrieval:** Performs multi-page real-time Wikipedia searches and retrieves verbalized Wikidata triples to cover homonymous topics and rare entities. Passages are reranked using GTR or BM25.
3. **Fact Verification:** Independently verifies each atomic fact against the retrieved context using a lightweight LLM (e.g., Llama-3-8B-Instruct) accelerated by `vLLM`.

*(📌 **Tip:** Add `assets/pipeline.png` here referencing Figure 1 from your paper)*

---


## 📦 Installation

### Option 1: Quick Start (For Users)
Install the latest stable release directly from PyPI:
```bash
pip install factowl
```

### Option 2: Development Setup (For Contributors & Researchers)
To run the full pipeline, including local Wikipedia dump indexing and Jupyter notebooks:

```bash
# 1. Clone the repository
git clone https://github.com/s-nlp/factowl.git
cd factowl

# 2. Create and activate the conda environment
conda env create -f environment.yml  # or use your factowl.yml
conda activate factowl

# 3. Install the package in editable mode
pip install -e .

# 4. Download required NLP models and data
python -m spacy download en_core_web_sm
python factowl/download_data.py
---------


# FactOwl: Blazingly fast and modern factchecker

## Usage Examples

1. [Fact generation and verification using pre-computed contexts](https://github.com/s-nlp/factowl/blob/main/examples/eval_no_retrieve_ridic_hf.ipynb) only on RiDiC dataset

2. [Fact generation and verification on FactScore dataset](https://github.com/s-nlp/factowl/blob/main/examples/eval_factowl_factscore_data.ipynb) usign either fixed Wikipedia dump or Wikipedia search for context retrieval

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


## 🚀 Quick Start

Evaluate your LLM generations via the Command Line Interface (CLI). FactOWL uses `vLLM` under the hood for maximum throughput.

```bash
python -m factowl --input_file data/sample_input.json --retriever wikipedia_search
```

### 📄 Input Data Format
Provide your data in a structured JSON format. FactOWL handles batch processing automatically:

```json
{
  "topics": {
    "topic_01": "William Post (lottery winner)",
    "topic_02": "Lanny Flaherty"
  },
  "generations": {
    "topic_01": "William Post (1949-1986) was an American lottery winner who won $16.2 million...",
    "topic_02": "Lanny Flaherty is an American actor born on December 18, 1949..."
  }
}
```

---

## 🧪 Examples & Notebooks

Explore our comprehensive Jupyter notebooks to understand the pipeline internals, experiment with context retrieval, and apply custom fact filtration rules.

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **FactScore Evaluation** | End-to-end fact generation and verification on the FActScore dataset using either a fixed Wiki dump or live search. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/s-nlp/factowl/blob/main/examples/eval_factowl_factscore_data.ipynb) |
| **Pre-computed Contexts** | Evaluate facts using pre-retrieved contexts (RiDiC dataset). Bypasses the retrieval step for isolated verification testing. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/s-nlp/factowl/blob/main/examples/eval_no_retrieve_ridic_hf.ipynb) |
| **Fact Filtration** | Advanced LLM and rule-based filtering of extracted atomic facts to remove noise and irrelevant claims. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/s-nlp/factowl/blob/main/examples/filter_facts.ipynb) |

---

## 📊 Hardware Requirements

FactOWL is designed to be highly accessible. While the original paper experiments were conducted on `3x NVIDIA RTX-3090 (24GB)`, the core verification pipeline utilizing **Llama-3-8B** and `vLLM` can comfortably run on a **single consumer GPU (e.g., RTX 3060/4070 with 12GB+ VRAM)** or Apple Silicon via MLX/Ollama integrations.

---

## 📚 Citation

If you use FactOWL in your research, demo, or production systems, please cite our SIGIR 2026 paper:

```bibtex
## 🚀 Quick Start

Evaluate your LLM generations via the Command Line Interface (CLI). FactOWL uses `vLLM` under the hood for maximum throughput.

```bash
python -m factowl --input_file data/sample_input.json --retriever wikipedia_search
```

### 📄 Input Data Format
Provide your data in a structured JSON format. FactOWL handles batch processing automatically:

```json
{
  "topics": {
    "topic_01": "William Post (lottery winner)",
    "topic_02": "Lanny Flaherty"
  },
  "generations": {
    "topic_01": "William Post (1949-1986) was an American lottery winner who won $16.2 million...",
    "topic_02": "Lanny Flaherty is an American actor born on December 18, 1949..."
  }
}
```

---

## 🧪 Examples & Notebooks

Explore our comprehensive Jupyter notebooks to understand the pipeline internals, experiment with context retrieval, and apply custom fact filtration rules.

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **FactScore Evaluation** | End-to-end fact generation and verification on the FActScore dataset using either a fixed Wiki dump or live search. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/s-nlp/factowl/blob/main/examples/eval_factowl_factscore_data.ipynb) |
| **Pre-computed Contexts** | Evaluate facts using pre-retrieved contexts (RiDiC dataset). Bypasses the retrieval step for isolated verification testing. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/s-nlp/factowl/blob/main/examples/eval_no_retrieve_ridic_hf.ipynb) |
| **Fact Filtration** | Advanced LLM and rule-based filtering of extracted atomic facts to remove noise and irrelevant claims. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/s-nlp/factowl/blob/main/examples/filter_facts.ipynb) |

---

## 📊 Hardware Requirements

FactOWL is designed to be highly accessible. While the original paper experiments were conducted on `3x NVIDIA RTX-3090 (24GB)`, the core verification pipeline utilizing **Llama-3-8B** and `vLLM` can comfortably run on a **single consumer GPU (e.g., RTX 3060/4070 with 12GB+ VRAM)** or Apple Silicon via MLX/Ollama integrations.

---

## 📚 Citation

If you use FactOWL in your research, demo, or production systems, please cite our SIGIR 2026 paper:

```bibtex
@inproceedings{10.1145/3805712.3808373,
author = {Sakhovskiy, Andrey and Sushko, Nikita and Marina, Maria and Konovalov, Vasily and Tutubalina, Elena and Panchenko, Alexander and Braslavski, Pavel},
title = {FactOWL: A Cost-Efficient Tool for Long-Form Factuality Evaluation},
year = {2026},
isbn = {9798400725999},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805712.3808373},
doi = {10.1145/3805712.3808373},
abstract = {The presence of factual hallucinations in large language model (LLM) generations drives the development of retrieval-augmented factuality evaluation tools. However, existing implementations suffer from slow inference, outdated knowledge bases, and the use of paid search and LLM APIs, which hinders research and practical applications. To fill this gap, we propose FactOWL, a FActScore-based Factuality evaluation tool which adopts an Open LLM and real-time Wikipedia search for evaluation of Long-form LLM responses. FactOWL effectively addresses unreliable evidence, incomplete contexts, and entity ambiguity by performing multi-source aggregation over Wikipedia pages and Wikidata triples. FactOWL is at least 10\texttimes{} faster than the pioneering FActScore fact checking tool and its key modifications methods and is freely available on GitHub: https://github.com/s-nlp/factowl.},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {5209–5214},
numpages = {6},
keywords = {llm factuality, factuality evaluation tool, atomic fact generation, fact verification, wikipedia, wikidata, evidence retrieval},
location = {Australia},
series = {SIGIR '26}
}
```

---

## 🙏 Acknowledgments

We pay our respects to the traditional owners of the lands where SIGIR 2026 was hosted, the peoples of the Woi Wurrung and Boon Wurrung language groups of the eastern Kulin Nation. We pay our respects to their Elders past and present, and extend that respect to all Aboriginal and Torres Strait Islander peoples today and their continuing connection to land, sea, sky, and community.
```

