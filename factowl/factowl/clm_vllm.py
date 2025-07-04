# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from factowl.atomic_facts_sped_up_vllm import VLLMGenerator
from vllm import SamplingParams
from transformers import AutoTokenizer

DEFAULT_VERIFICATION_SYSTEM_PROMPT = "You are an expert fact extraction and verification assistant. " \
                                     "You are given an atomic fact and a list of textual passages. Your task is to determine whether the atomic fact " \
                                     "is True or False  based solely on the information in the passages.\n" \
                                     "Instructions:\n" \
                                     "1. Check if any of the passages directly support the atomic fact.\n" \
                                     "2. Output 'True' if at least one passage supports the fact even if another passage contradicts the fact.\n" \
                                     "3. Output 'False' if no passage  supports the fact.\n" \
                                     "4. Do not include any additional information and explanations, you must only answer 'True' or 'False'."


DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE = """
Passages:
<passages>
Atomic Fact: <atomic_fact>
True or False? 
"""

DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE_WITH_TOPIC = """
Passages:
<passages>
Atomic Fact's topic/topics: <fact_topic>
Atomic Fact: <atomic_fact>
True or False? 
"""


class FactVerificatorSpedUpVLLM(object):
    def __init__(self, vllm_model, model_name, is_bio=False, debug=False,
                 system_prompt=None, query_prompt_template=None, max_tokens: int = 1024, temperature: float = 0.,
                 context_type="db"):
        assert context_type in ("db", "wikipedia_api")
        self.is_bio = is_bio
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vllm = VLLMGenerator(vllm_model=vllm_model, model_name=model_name, debug=debug, temperature=temperature,
                                  max_tokens=max_tokens)

        self.debug = debug
        if system_prompt is None:
            system_prompt = DEFAULT_VERIFICATION_SYSTEM_PROMPT
        self.system_prompt = system_prompt
        if query_prompt_template is None:
            query_prompt_template = DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE
        self.query_prompt_template = query_prompt_template
        self.max_new_tokens = max_tokens
        self.temperature = temperature
        self.context_type = context_type

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,  # deterministic
            stop=[self.tokenizer.eos_token]
        )

    def create_messages(self, query, passages, topic=None):
        assert self.system_prompt is not None
        assert self.query_prompt_template is not None

        if topic is not None:
            query_prompt_template = DEFAULT_FACT_VERIFICATION_QUERY_TEMPLATE_WITH_TOPIC
            query_prompt = query_prompt_template.replace('<atomic_fact>', query)
            if isinstance(topic, list):
                topic_s = ','.join(sorted(topic))
            elif isinstance(topic, str):
                topic_s = topic
            else:
                raise ValueError(f"Unsupported topic type {type(topic)} for {topic}")
            query_prompt = query_prompt.replace("<fact_topic>", topic_s)

        else:
            query_prompt_template = self.query_prompt_template
            query_prompt = query_prompt_template.replace('<atomic_fact>', query)

        context_s = ""
        for i, psg in enumerate(passages):
            s = "Title: {}\nText: {}\n\n".format(psg["title"],
                                                 psg["text"].replace("<s>", "").replace("</s>", ""))
            context_s += s
        query_prompt = query_prompt.replace("<passages>", context_s)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query_prompt}
        ]

        return messages

    def generate(self, prompts):
        return self.vllm.generate(prompts)
