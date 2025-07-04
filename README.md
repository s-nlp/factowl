
```bash
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
python local_factscore/factowl/download_data.py
```

*   **Python 3.7+**
*   **API Keys:**
    *   **OpenAI API Key:** You need an OpenAI API key to use the LLM for fact extraction. Set the `OPENAI_API_KEY` environment variable.
    *   **OpenAI API Base URL:** You may need to set the `OPENAI_API_BASE` environment variable if you are not using the standard OpenAI API endpoint.
*   **Install the required Python packages:**

    ```bash
    pip install langchain langchain-huggingface scikit-learn numpy sentence-transformers wikipedia pandas python-dotenv openai
    ```

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required Python packages (see "Requirements" above).

3.  Create a `.env` file in the project root and set your OpenAI API key:

    ```
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    OPENAI_API_BASE=YOUR_OPENAI_API_BASE_URL (optional)
    ```

## Usage

Run the `wikidata.py` script from the command line, with optional arguments to customize the behavior:

```bash
python wikidata.py --input-file <input_file.csv> --output-file <output_file.json> --max-workers <num_workers> --openai-model <openai_model_name> --embedding-model <embedding_model_name>
```

### Arguments

*   `--input-file`: Path to the input CSV file containing river data (default: `rivers.csv`). The CSV should contain columns named `itemLabel` (river name) and `item` (Wikidata link).
*   `--output-file`: Path to the output JSON file. If not specified, the output is printed to standard output.
*   `--max-workers`: Maximum number of threads to use for parallel fact extraction (default: 20).
*   `--openai-model`: Name of the OpenAI model to use for fact extraction (default: `mistralai/mistral-small-3.1-24b-instruct`).
*   `--embedding-model`: Name of the Hugging Face model to use for embeddings and deduplication (default: `all-MiniLM-L6-v2`).

### Example

To extract facts about rivers from `my_rivers.csv`, use the `gpt-3.5-turbo` model, and save the output to `river_facts.json`, using 30 threads, run:

```bash
python wikidata.py --input-file my_rivers.csv --output-file river_facts.json --max-workers 30 --openai-model gpt-3.5-turbo --embedding-model all-MiniLM-L6-v2
```

## File Structure

*   `wikidata.py`: The main script for extracting river facts.
*   `rivers.csv`: (Example) CSV file containing river names and Wikidata links.
*   `src/`:
    *   `prompts.py`: Defines the prompt used for fact extraction with the LLM.
    *   `utils.py`: Contains utility functions, such as the `call_openai` function for interacting with the OpenAI API.
*   `README.md`: This file, providing an overview of the project.

## Notes

*   The script processes rivers sequentially from the input CSV file.
*   The quality of the extracted facts depends on the quality of the Wikipedia articles and the capabilities of the chosen LLM.
*   Consider the rate limits and costs associated with the OpenAI API when running the script with a large input file.
*   The `auto_suggest=False` argument is passed to `wikipedia.page()` to prevent the Wikipedia library from automatically correcting the river name. This is to avoid issues with rivers that have unusual names.

## License

[Choose a license and add it here]
# factowl
