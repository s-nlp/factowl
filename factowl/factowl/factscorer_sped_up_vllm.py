import logging
import os

import numpy as np
import pandas as pd
from factowl.abstain_detection import is_response_abstained
from factowl.atomic_facts_sped_up_vllm import AtomicFactGeneratorSpedUpVLLM
from factowl.clm_vllm import FactVerificatorSpedUpVLLM
from factowl.npm import NPM
from factowl.retrieval import DocDB, Retrieval
from tqdm import tqdm


class FactScorerSpedUpVLLM(object):
    def __init__(self,
                 vllm_model,
                 model_name="retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 abstain_detection_type=None,
                 batch_size=256,
                 is_bio: bool = False,
                 debug: bool = False,
                 atomic_facts_cache_dir: str = None,
                 dump_every_int: int = 10,
                 fact_generator_max_tokens: int = 2048,
                 verifier_temperature: float = 0.,
                 verifier_max_tokens: int = 8,
                 retrieval_device: str = "cuda",
                 context_retrieval_type: str = "gtr-t5-large",
                 npm_retrieval_type: str = "bm25",
                 context_type: str = "db",
                 context_num_pages: int = 1,
                 n_npm_contexts: int = 3,
                 num_supporting_contexts: int = 10,
                 precomputed_passages=None,
                 concat_topic=False
                 ):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm",
                              "retrieval+ChatGPT+npm"]
        assert context_type in ("db", "wikipedia_api")
        assert context_retrieval_type in ("gtr-t5-large", "bm25")
        assert npm_retrieval_type in ("gtr-t5-large", "bm25")
        page_search_mode = "single" if context_num_pages == 1 else "multi"
        assert page_search_mode in ("single", "multi")

        logging.info(f"FactScore is using context retrieval type: {context_type}")
        self.model_name = model_name
        self.vllm_model = vllm_model

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size  # batch size for retrieval
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        llm_dir_or_name = vllm_model.llm_engine.model_config.tokenizer
        self.llm_dir_or_name = llm_dir_or_name

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.debug = debug
        self.is_bio = is_bio

        llm_base_name = os.path.basename(llm_dir_or_name)
        self.llm_base_name = llm_base_name
        self.llm = None
        self.tokenizer = None

        self.atomic_facts_cache_dir = atomic_facts_cache_dir
        self.dump_every_int = dump_every_int
        self.retrieval_device = retrieval_device

        self.fact_generator_max_tokens = fact_generator_max_tokens
        self.verifier_temperature = verifier_temperature
        self.verifier_max_tokens = verifier_max_tokens
        self.vllm_verifier = FactVerificatorSpedUpVLLM(vllm_model, model_name=self.llm_dir_or_name, debug=debug,
                                                       temperature=verifier_temperature,
                                                       max_tokens=verifier_max_tokens,
                                                       context_type=context_type)
        self.cxt_type = context_type
        self.context_retrieval_type = context_retrieval_type
        self.npm_retrieval_type = npm_retrieval_type
        self.page_search_mode = page_search_mode
        self.cxt_n_pages = context_num_pages
        self.n_support_cxt = num_supporting_contexts
        self.n_npm_cxt = n_npm_contexts
        self.precomputed_passages = precomputed_passages
        self.extr_ps = ''
        if self.precomputed_passages is not None:
            self.extr_ps = '-extra'
        self.concat_topic = concat_topic
        # self.torch_compile = torch_compile

    def register_knowledge_source(self, name="enwiki-20240401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir,
                                  f"retrieval-{name}-{self.cxt_type}-p{self.cxt_n_pages}-c{self.n_support_cxt}{self.extr_ps}.json")
        embed_cache_path = os.path.join(self.cache_dir,
                                        f"retrieval-{name}-{self.cxt_type}-p{self.cxt_n_pages}-c{self.n_support_cxt}{self.extr_ps}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, "bm25",
                                         batch_size=self.batch_size,
                                         device=self.retrieval_device,
                                         context_type=self.cxt_type,
                                         page_search_mode=self.page_search_mode,
                                         context_num_pages=self.cxt_n_pages)
        cache_file = os.path.join(self.cache_dir,
                                  f"npm-{name}-{self.cxt_type}-p{self.cxt_n_pages}-c{self.n_support_cxt}{self.extr_ps}.pkl")
        self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, self.npm_retrieval_type,
                                       device=self.retrieval_device, context_type=self.cxt_type,
                                       page_search_mode=self.page_search_mode,
                                       context_num_pages=self.cxt_n_pages),
                             "npm-single",
                             cache_file=cache_file,
                             device=self.retrieval_device,
                             context_type=self.cxt_type)
        
    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  all_atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if all_atomic_facts is not None:
            assert len(topics) == len(all_atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGeneratorSpedUpVLLM(demon_dir=os.path.join(self.data_dir, "demos"),
                                                                  vllm_model=self.vllm_model,
                                                                  model_name=self.llm_dir_or_name,
                                                                  max_tokens=self.fact_generator_max_tokens,
                                                                  is_bio=self.is_bio,
                                                                  debug=self.debug)
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics) == type(generations) == str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics) == type(generations) == list, "`topics` and `generations` should be lists."
            assert len(topics) == len(generations), "`topics` and `generations` should have the same length"

        if verbose:
            topics = tqdm(topics)
        topic_labels = []

        batch_atomic_facts = []
        batch_topic_labels = []
        all_atomic_facts = []
        all_decisions = []
        for i, (topic, gen) in enumerate(zip(topics, generations)):
            # optionally, first detect if the response is abstained
            response_abstained = is_response_abstained(gen, self.abstain_detection_type)
            if response_abstained:
                all_atomic_facts.append(None)
                batch_atomic_facts.append(None)
                # all_decisions.append(None)
                topic_labels.append(topic)
                batch_topic_labels.append(topic)

            # continue only when the response is not abstained
            curr_afs, _ = self.af_generator.run(gen)
            curr_afs = set([fact for _, facts in curr_afs for fact in facts])

            if self.debug:
                logging.info(f"Topic: {topic}")
                logging.info(f"Generation (len: {len(gen)}): {gen[:200]} ... {gen[200:]}")
                logging.info(f"\tAtomic facts: {len(curr_afs)}")
                for af in curr_afs:
                    logging.info(f"\tAtomic fact: {af}")

            if len(curr_afs) == 0:
                all_atomic_facts.append(None)
                batch_atomic_facts.append(None)
                topic_labels.append(topic)
                batch_topic_labels.append(topic)
            else:
                all_atomic_facts.append(curr_afs)
                topic_labels.append(topic)
                batch_atomic_facts.append(curr_afs)
                batch_topic_labels.append(topic)
            if i > 0 and i % self.dump_every_int == 0:
                self.batch_verify_facts(batch_topic_labels=batch_topic_labels,
                                        batch_atomic_facts=batch_atomic_facts,
                                        all_decisions=all_decisions,
                                        knowledge_source=knowledge_source)
                c = sum(len(z) for z in all_decisions)
                decisions_fname = f"decisions_topics-{len(all_decisions)}-facts-{c}.tsv"
                self.aggregate_verified_fact_scores(all_decisions=all_decisions,
                                                    all_atomic_facts=all_atomic_facts,
                                                    gamma=gamma,
                                                    batch_atomic_facts=batch_atomic_facts,
                                                    batch_topic_labels=batch_topic_labels,
                                                    decisions_fname=decisions_fname)
        if len(batch_atomic_facts) > 0:
            self.batch_verify_facts(batch_topic_labels=batch_topic_labels,
                                    batch_atomic_facts=batch_atomic_facts,
                                    all_decisions=all_decisions,
                                    knowledge_source=knowledge_source)
        c = sum(len(z) for z in all_decisions)
        decisions_fname = f"decisions_topics-{len(all_decisions)}-facts-{c}.tsv"
        eval_dict = self.aggregate_verified_fact_scores(all_decisions=all_decisions,
                                                        all_atomic_facts=all_atomic_facts,
                                                        gamma=gamma,
                                                        batch_atomic_facts=batch_atomic_facts,
                                                        batch_topic_labels=batch_topic_labels,
                                                        decisions_fname=decisions_fname)

        return eval_dict

    def aggregate_verified_fact_scores(self, all_decisions, all_atomic_facts, gamma, batch_atomic_facts,
                                       batch_topic_labels, decisions_fname):

        eval_dict = calculate_score_from_decisions(all_atomic_facts=all_atomic_facts,
                                                   decisions=all_decisions, gamma=gamma)
        all_decisions_df = pd.DataFrame(eval_dict["decisions"])
        decisions_path = os.path.join(self.atomic_facts_cache_dir, decisions_fname)
        d = os.path.dirname(decisions_path)
        if not os.path.exists(d):
            os.makedirs(d)
        all_decisions_df.to_csv(decisions_path, sep='\t', index=False)

        scores_path = os.path.join(self.atomic_facts_cache_dir, f"scores/eval_scores.tsv")
        d = os.path.dirname(scores_path)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(scores_path, 'a+', encoding="utf-8") as out_file:
            if eval_dict.get("init_score") is not None:
                isc = eval_dict["init_score"]
            else:
                isc = ''
            sc = eval_dict["score"]
            rr = eval_dict["respond_ratio"]
            nfpr = eval_dict["num_facts_per_response"]
            out_file.write(f"{len(all_decisions)}\t{isc}\t{sc}\t{rr}\t{nfpr}\n")

        return eval_dict

    def batch_verify_facts(self, batch_topic_labels, batch_atomic_facts, all_decisions,
                           knowledge_source):
        assert len(batch_topic_labels) == len(batch_atomic_facts)
        old_len = len(all_decisions)
        for t, afs in tqdm(zip(batch_topic_labels, batch_atomic_facts), total=min(len(batch_topic_labels), len(batch_atomic_facts))):
            if afs is None or len(afs) == 0:
                topic_decisions = [{"topic": t, "atom": None, "is_supported": False}, ]
            else:
                topic_decisions = self._get_score_vllm(topic=t, atomic_facts=afs,
                                                       knowledge_source=knowledge_source)
            all_decisions.append(topic_decisions)
        new_len = len(all_decisions)
        try:
            assert new_len - old_len == len(batch_atomic_facts)
        except AssertionError:
            logging.warning(f'Some of the generations were not checked: {new_len=}, {old_len=}')
        batch_atomic_facts.clear()
        batch_topic_labels.clear()

    def _get_score_vllm(self, topic, atomic_facts, knowledge_source):
        decisions = []
        total_words = 0
        prompts = []

        for atom in atomic_facts:
            atom = atom.strip()
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=self.n_support_cxt)
            # if isinstance(topic, str):
            #     passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=self.n_support_cxt)
            # elif isinstance(topic, list):
            #     passages = []
            #     for t in topic:
            #         psgs = self.retrieval[knowledge_source].get_passages(t, atom, k=self.n_support_cxt)
            #         passages.extend(psgs)
            # else:
            #     raise RuntimeError(f"Unsupported topic type: {type(topic)} for topic {topic}")
            if self.precomputed_passages is not None:
                extra_psgs = self.precomputed_passages.get(topic)
                if extra_psgs is not None:
                    passages.extend(extra_psgs)
            if self.debug:
                logging.info(f"FACT EVALUATION CONTEXT PASSAGES:\n{passages}\n--")

            if self.concat_topic:
                verification_prompt = self.vllm_verifier.create_messages(query=atom, passages=passages, topic=topic)
            else:
                verification_prompt = self.vllm_verifier.create_messages(query=atom, passages=passages, )
            prompts.append(verification_prompt)

        prompts = [
            self.vllm_verifier.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompts
        ]
        if self.debug:
            logging.info(f"Atomic facts prompt:\n{prompts}")
        outputs = self.vllm_verifier.generate(prompts)
        gen_texts = [o.outputs[0].text for o in outputs]
        assert len(gen_texts) == len(atomic_facts)
        for gen, atom in zip(gen_texts, atomic_facts):
            is_supported = generation_get_verification(gen)

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom, k=self.n_npm_cxt)
                is_supported = npprob > 0.3

            decisions.append({"topic": topic, "atom": atom, "is_supported": is_supported})
        assert len(gen_texts) == len(atomic_facts) == len(decisions)
        if self.debug:
            logging.info("Verifying atomic facts....")
            for at, g, dec_d in zip(atomic_facts, gen_texts, decisions):
                logging.info(f"{at} - {g} - {dec_d['is_supported']}")

        return decisions


def generation_get_verification(gen):
    gen = gen.strip().strip('.').strip().lower()
    if gen == "true":
        return True
    elif gen == "false":
        return False
    else:
        return False


def calculate_score_from_decisions(all_atomic_facts, decisions, gamma):
    scores = []
    init_scores = []
    flat_decisions = []
    assert len(decisions) == len(all_atomic_facts)
    for decision, atomic_facts in zip(decisions, all_atomic_facts):

        if atomic_facts is None:
            continue
        score = np.mean([d["is_supported"] for d in decision])
        if gamma:
            init_scores.append(score)
            penalty = 1.0 if len(atomic_facts) > gamma else np.exp(1 - gamma / len(atomic_facts))
            score = penalty * score
        scores.append(score)
        flat_decisions.extend(decision)

    respond_ratio = np.mean([facts is not None for facts in all_atomic_facts])

    out = {"score": np.mean(scores),
           "respond_ratio": respond_ratio,
           "decisions": flat_decisions,
           "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

    if gamma:
        out["init_score"] = np.mean(init_scores)

    return out

