#!/usr/bin/env python3
"""
CLI interface for factowl package.
Usage: python -m factowl --input data/ChatGPT_FactScore.json --output res.json
"""

import argparse
import json
import os
import pandas as pd
import pathlib
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to import factowl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factowl.factscorer_sped_up_vllm import FactScorerSpedUpVLLM
from factowl.io import save_predictions, save_eval_results
from vllm import LLM


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_input_data(input_path, model):
    """Load input data from JSON file with topics and generations."""
    try:
        logger.info(f"Loading input data from {input_path}")
        
        # Load JSON with pandas to handle nested structure
        data = json.loads(pathlib.Path(input_path).read_text())
        
        # Extract topics and generations from nested dictionaries
        topics = [_['title_en'] for _ in data]
        generations = [_[model] for _ in data]
        
        return topics, generations
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise


def setup_factscorer(model_name="retrieval+llama", 
                    cache_dir="./cache",
                    data_dir="./wikipedia_dumps",
                    context_type="wikipedia_api",
                    context_num_pages=1,
                    num_supporting_contexts=10):
    """Setup FactScorer with default parameters."""
    
    logger.info("Setting up FactScorer...")
    
    # Setup vLLM model
    vllm_model = LLM(
        model='meta-llama/Meta-Llama-3-8B-Instruct',
    )
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    fs = FactScorerSpedUpVLLM(
        vllm_model=vllm_model,
        model_name=model_name,
        cache_dir=cache_dir,
        data_dir=data_dir,
        context_type=context_type,
        context_num_pages=context_num_pages,
        num_supporting_contexts=num_supporting_contexts,
        debug=False
    )
    
    return fs


def run_factscorer(topics, generations, fs, output_path, knowledge_source="enwiki-20230401"):
    """Run factscorer on the provided topics and generations."""
    
    logger.info("Registering knowledge source...")
    fs.register_knowledge_source(knowledge_source)
    
    logger.info("Running fact verification...")
    results = fs.get_score(
        topics=topics,
        generations=generations,
        gamma=10,
        knowledge_source=knowledge_source,
        verbose=True
    )
    
    logger.info(f"Score: {results['score']}")
    logger.info(f"Respond ratio: {results['respond_ratio']}")
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also save predictions in TSV format
    tsv_path = output_path.replace('.json', '_predictions.tsv')
    save_predictions(results, tsv_path)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Predictions saved to {tsv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FactOwl CLI - Fact verification tool')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input JSON file with topics and generations')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSON file for results')
    parser.add_argument('--model', default='retrieval+llama',
                       choices=['retrieval+llama', 'retrieval+ChatGPT', 'npm'],
                       help='Model to use for fact verification')
    parser.add_argument('--cache-dir', default='./cache',
                       help='Directory for cache files')
    parser.add_argument('--data-dir', default='./wikipedia_dumps',
                       help='Directory for Wikipedia data')
    parser.add_argument('--knowledge-source', default='enwiki-20230401',
                       help='Knowledge source to use')
    parser.add_argument('--context-type', default='wikipedia_api',
                       choices=['wikipedia_api', 'db'],
                       help='Type of context retrieval')
    parser.add_argument('--context-pages', type=int, default=1,
                       help='Number of context pages to retrieve')
    parser.add_argument('--context-passages', type=int, default=10,
                       help='Number of supporting passages per fact')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    for model in ['llama3.1:8b', 'qwen2.5:7b']:
        topics, generations = load_input_data(args.input, model)
        output_path = args.output.split('.')[0] + f'_{model}' + args.output.split('.')[1]
        
        if not topics or not generations:
            logger.error("No valid topics/generations found in input file")
            sys.exit(1)
        
        # Setup FactScorer
        fs = setup_factscorer(
            model_name=args.model,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            context_type=args.context_type,
            context_num_pages=args.context_pages,
            num_supporting_contexts=args.context_passages
        )
        
        # Run factscorer
        results = run_factscorer(
            topics=topics,
            generations=generations,
            fs=fs,
            output_path=output_path,
            knowledge_source=args.knowledge_source
        )
        
        logger.info("Fact verification completed successfully!")
        


if __name__ == "__main__":
    main()

