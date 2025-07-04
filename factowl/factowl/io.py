import json
import numpy as np
import os
import pandas as pd
from nltk import sent_tokenize


def load_json_generations(p):
    topics = []
    gens = []
    with open(p, 'r', encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            t = doc["topic"]
            g = doc["output"]

            topics.append(t)
            gens.append(g)
    return topics, gens


def load_json_atomic_facts(p):
    topics = []
    gens = []
    with open(p, 'r', encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            t = doc["topic"]
            g = doc["output"]

            topics.append(t)
            gens.append(g)
    return topics, gens


def save_predictions(eval_dict, save_path):
    d = os.path.dirname(save_path)
    if not os.path.exists(d):
        os.makedirs(d)
    decisions = eval_dict["decisions"]
    samples = []
    dedup_samples = []
    seen_atoms = set()
    for sample_id, fact_dict in enumerate(decisions):
        if fact_dict is None:
            continue
        seen_atoms.clear()
        # for fact_dict in dec:
        atom = fact_dict["atom"]
        sup = fact_dict["is_supported"]
        assert sup == np.True_ or sup == np.False_
        label = 1 if sup == np.True_ else 0
        d = {
            "sample_id": sample_id,
            "topic": fact_dict["topic"],
            "atom": fact_dict["atom"],
            "is_supported": fact_dict["is_supported"],
            "label": label
        }
        samples.append(d)
        if atom not in seen_atoms:
            dedup_samples.append(d)

            seen_atoms.add(atom)
    score = sum(x["label"] for x in samples) / len(samples)
    dedup_score = sum(x["label"] for x in dedup_samples) / len(dedup_samples)

    num_facts = sum(len(x) for x in samples)
    num_dedup_facts = sum(len(x) for x in dedup_samples)

    print(f"Mean num facts: {num_facts / len(decisions)}")
    print(f"Mean unique num facts:  {num_dedup_facts / len(decisions)}")

    print(f"Mean score: {score}")
    print(f"Mean deduplicated score: {dedup_score}")
    if eval_dict.get('init_score') is not None:
        print(f"init_score: {eval_dict['init_score']}")

    print(f"Method's score: {eval_dict['score']}")
    eval_dict["mean_custom_score"] = score
    eval_dict["mean_custom_deduplicated_score"] = dedup_score

    df = pd.DataFrame(samples)
    print(f"Created DataFrame. Size: {df.shape}, Columns: {df.columns}")
    out_dir = os.path.dirname(save_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(save_path, sep='\t', index=False)


def save_eval_results(eval_dict, save_path):
    d = os.path.dirname(save_path)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(save_path, 'w', encoding="utf-8") as f:
        score = eval_dict["score"]
        respond_ratio = eval_dict["respond_ratio"]
        init_score = eval_dict["init_score"]
        num_facts_per_response = eval_dict["num_facts_per_response"]

        mcdc = eval_dict["mean_custom_deduplicated_score"]
        mcc = eval_dict["mean_custom_score"]

        f.write(f"score\t{score}\n")
        f.write(f"init_score\t{init_score}\n")
        f.write(f"respond_ratio\t{respond_ratio}\n")
        f.write(f"num_facts_per_response\t{num_facts_per_response}\n")

        f.write(f"mean_custom_score\t{mcc}\n")
        f.write(f"mean_custom_deduplicated_score\t{mcdc}\n")


def json_docs2title_short_passages(docs, max_tokens=256):
    topic2psgs = {}
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    for d in docs:
        to = d["topic"]
        psgs = d["passages"]
        passages = []
        for ps_d in psgs:
            ti = ps_d["title"]
            te = ps_d["text"]
            cur_paras = []
            chunk_size = 0
            sentences = set(sent_tokenize(te))

            for sent in sentences:
                tokens = tokenizer(sent)["input_ids"]
                if chunk_size + len(tokens) > max_tokens:
                    text = ' '.join(cur_paras)
                    passages.append({"title": ti.strip(), "text": text})

                    cur_paras.clear()
                    chunk_size = 0
                cur_paras.append(sent)
                chunk_size += len(tokens)
            if len(cur_paras) > 0:
                text = ' '.join(cur_paras)
                passages.append({"title": ti.strip(), "text": text})

        topic2psgs[to] = passages
    return topic2psgs


def json_docs2title_text_passages(docs):
    topic2psgs = {}
    for d in docs:
        to = d["topic"]
        psgs = d["passages"]
        for ps_d in psgs:
            ti = ps_d["title"]
            te = ps_d["text"]
            ps_d["title"] = f"{to}. {ti}"
        topic2psgs[to] = psgs
    return topic2psgs
