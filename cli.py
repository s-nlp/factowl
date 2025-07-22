import argparse
import json
import os
import pathlib

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from factowl import FactScorerSpedUpVLLM as FactScorer
from factowl.io import save_predictions, load_simple_json


def parse_args():
    parser = argparse.ArgumentParser(description="FactOwl example script. Works only on CSV files.")

    parser.add_argument('--input_file', type=str, help='Path to the input file.')
    parser.add_argument('--cnp', type=int, default=1, help='Number of context pages.')
    parser.add_argument('--nsp', type=int, default=10, help='Number of relevant passages.')

    args = parser.parse_args()

    return args


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["VLLM_LOG_LEVEL"] = "WARNING"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    vllm_model = LLM(
        model='meta-llama/Meta-Llama-3-8B-Instruct',
    )

    json_p = os.path.join(args.input_file)
    atomic_facts_cache_dir = f"./cache/{args.input_file.split('.')[0]}-wikipedia_api-p{args.cnp}-c{args.nsp}/"
    print(f"{atomic_facts_cache_dir=}")

    fs = FactScorer(
        model_name='retrieval+llama',
        data_dir='./data/',
        vllm_model=vllm_model,
        atomic_facts_cache_dir=atomic_facts_cache_dir,
        dump_every_int=20,
        cache_dir='./cachedir/',
        abstain_detection_type='generic',
        is_bio=False,
        retrieval_device="cuda:1",
        context_type='wikipedia_api',
        context_num_pages=args.cnp,
        num_supporting_contexts=args.nsp,
        debug=True,
    )

    topics, generations = load_simple_json(json_p)    
    all_decisions = []

    fs.register_knowledge_source('enwiki-20230401', db_path='./data/enwiki-20230401.db', data_path=args.input_file)
    out = fs.get_score(topics, generations)

    print(f'Score: {out["score"]}\nRespond ratio: {out["respond_ratio"]}')
    pathlib.Path(f"{args.input_file.split('.')[0]}_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))



if __name__ == '__main__':
    args = parse_args()
    main(args)
