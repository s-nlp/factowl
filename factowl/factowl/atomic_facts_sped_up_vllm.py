import json
import logging
import os
import re
import string

import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Ensure required models are downloaded
nltk.download("punkt")

DEFAULT_ATOMIZATION_PROMPT = "You are an expert fact extraction and verification assistant. " \
                 "Please read the following text carefully and break it down into distinct, independent facts. " \
                 "For each fact, disambiguate it to ensure clarity and precision (e.g., replace ambiguous prepositions). " \
                 "Each fact should be written on its own line. " \
                 "Each line must start with a hyphen and space ('- '). " \
                 "Do not include any additional explanation or formatting - just the list of facts if there are any."


class VLLMGenerator:
    def __init__(self, vllm_model, model_name,
                 debug=False, temperature: float = 0., max_tokens: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        # vLLM uses its own sampling config
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,  # deterministic
            stop=[self.tokenizer.eos_token]
        )
        self.debug = debug

        # Load model with tensor parallelism if multi-GPU
        self.model = vllm_model

    def generate(self, prompts):
        return self.model.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)


class AtomicFactGeneratorSpedUpVLLM(object):
    def __init__(self, demon_dir, vllm_model, model_name, is_bio=False, debug=False,
                 system_prompt=None, max_tokens: int = 2048, temperature: float = 0.):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = is_bio
        # self.is_bio = True
        self.demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")
        self.vllm = VLLMGenerator(vllm_model=vllm_model, model_name=model_name, debug=debug, temperature=temperature,
                                  max_tokens=max_tokens)
        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)
        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.debug = debug
        if system_prompt is None:
            system_prompt = DEFAULT_ATOMIZATION_PROMPT
        self.system_prompt = system_prompt
        self.max_new_tokens = max_tokens
        self.temperature = temperature

    def create_messages(self, example_queries, example_outputs, new_query):
        assert self.system_prompt is not None
        assert len(example_queries) == len(example_outputs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            # {"role": "user", "content": para}
        ]
        for ex_q, ex_out in zip(example_queries, example_outputs):
            q_d = {"role": "user", "content": ex_q}
            out_d = {"role": "assistant", "content": ex_out}

            messages.append(q_d)
            messages.append(out_d)
        new_q_d = {"role": "user", "content": new_query}
        messages.append(new_q_d)

        return messages

    def run(self, generation, cost_estimate=None):
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = []
        for g in generation.split("\n"):
            pars = [para.strip() for para in g.split("    ") if len(para.strip()) > 0]
            paragraphs.extend(pars)
        if self.debug:
            logging.info(f"Splitting generation (len: {len(generation)}): {generation[:200]} ... {generation[200:]} ")
            for p in paragraphs[:3]:
                logging.info(f"\tSplitted paragraph: {p}")
        return self.get_atomic_facts_from_paragraph(paragraphs, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(self, paragraphs, cost_estimate=None):
        # sentences = []
        para_breaks = []

        atoms_or_estimate = self.get_init_atomic_facts_from_paragraphs(paragraphs, cost_estimate=cost_estimate)
        """
        [sent for i, sent in enumerate(sentences) if not (not self.is_bio and (
            (i == 0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or 
            (i == len(sentences) - 1 and (
                sent.startswith("Please") or 
                sent.startswith("I hope") or 
                sent.startswith("Here are")
            ))
        ))]
        """
        # TODO: CHANGE PROMPT AND PROCESS NO FACTS CASE?
        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate
        atomic_facts_pairs = []
        for i, para in enumerate(paragraphs):
            if self.is_bio and para.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((para, []))
            else:
                atomic_facts_pairs.append((para, atoms[para]))
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)
        return atomic_facts_pairs, para_breaks

    def get_init_atomic_facts_from_paragraphs(self, paragraphs, cost_estimate=None):
        is_bio = self.is_bio
        demons = self.demons
        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        messages = []
        prompt_to_sent = {}
        atoms = {}
        for para in paragraphs:
            if para in atoms:
                continue
            top_machings = best_demos(para, self.bm25, list(demons.keys()), k)
            keys = set(list(demons.keys()) + top_machings)
            values = [demons[key] for key in keys]

            example_text = " ".join(keys)
            example_answer = ""
            for lst in values:
                example_answer += ''.join(f"- {x}\n" for x in lst)

            example_queries = [example_text, ]
            example_outputs = [example_answer, ]
            new_query = para

            prompt = self.create_messages(example_queries, example_outputs, new_query)

            messages.append(prompt)
            # prompt_to_sent[prompt] = para
        if cost_estimate:
            total_words_estimate = 0
            for prompt in messages:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.llm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            if self.debug:
                logging.info(f"Generating atomic facts....")

            prompts = [
                self.vllm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in messages
            ]
            if self.debug:
                logging.info(f"Atomic facts prompt:\n{prompts}")
            outputs = self.vllm.generate(prompts)
            gen_texts = [o.outputs[0].text for o in outputs]
            if self.debug:
                s_lst = [f"PROMPT:{x}\nOUTPUT:{y}\n" for x, y in zip(prompts, gen_texts)]
                nl = '\n--\n'
                logging.info(f"{nl.join(s_lst)}")
            assert len(gen_texts) == len(paragraphs)
            for p, gen_text in zip(paragraphs, gen_texts):

                afs = text_to_sentences(gen_text)
                atoms[p] = afs
                if self.debug:
                    logging.info(f"Splitting generation:\n{gen_text}")
                    logging.info(f"LLM output sentences:\n{afs}")
            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value
            return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# TODO: transform InstructGPT output into sentences
def text_to_sentences(text):
    if isinstance(text, tuple):
        raise RuntimeError(f"Error. text_to_sentences got input\n:{text}")
    # text = text.split('\n\n')[0]
    # sentences = text.split("- ")[1:]
    sentences = [x.strip('-').strip() for x in text.split('\n') if x.startswith('- ') and x.strip('-').strip() != '']
    # sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0:
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.'
    else:
        sentences = []
    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
MONTHS = [m.lower() for m in MONTHS]


def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False


def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True


def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)

    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)

    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities


def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.",
             "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split()) == 1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any(
                    [fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities == covered_entities
        except Exception:
            new_facts = facts  # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False


def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]


def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip()) > 0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [
                        curr_sentences[i] + " " + curr_sentences[i + 1]] + curr_sentences[i + 2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


def main():
    generator = AtomicFactGeneratorSpedUp("api.key", "demos", gpt3_cache_dir=None)
    atomic_facts, para_breaks = generator.run(
        "Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")

    print(atomic_facts)
    print(para_breaks)


if __name__ == "__main__":
    main()
