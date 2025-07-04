import json
import os
import pickle as pkl
import sqlite3
import time

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import RobertaTokenizer
from factowl.utils import retrieve_wikipedia_page, retrieve_multiple_wikipedia_pages

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print(f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text) == str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip()) > 0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset + MAX_LENGTH])
                            offset += MAX_LENGTH

                psgs = [tokenizer.decode(tokens) for tokens in passages if
                        np.sum([t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(
            results) == 1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results


class Retrieval(object):

    def __init__(self, db, cache_path, embed_cache_path, retrieval_type="gtr-t5-large", batch_size=None,
                 device="cuda", page_search_mode: str = "single", context_type=None, context_num_pages: int = 1):
        assert context_type in ("db", "wikipedia_api")
        assert page_search_mode in ("single", "multi")
        if page_search_mode == "single":
            assert context_num_pages == 1
        elif page_search_mode == "multi":
            assert context_num_pages > 1
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type == "bm25" or retrieval_type.startswith("gtr-")

        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0
        self.device = device
        self.context_type = context_type
        self.psg_tokenizer = None
        if context_type == "wikipedia_api":
            self.psg_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.page_search_mode = page_search_mode
        self.context_num_pages = context_num_pages

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type, device=self.device)
        encoder = encoder.to(self.device)
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def wikipedia_page2passages(self, doc, paragraph_max_tokens=MAX_LENGTH):
        passages = []
        if doc is None:
            return passages

        title = doc.title
        cont = doc.content
        lines = cont.split('\n')

        stop_sections = {"References", "See also", "External links", "Further reading", "Notes", "Citations", "Sources"}
        cur_section = ""
        cur_paras = []
        chunk_size = 0
        for line in lines:
            line = line.strip()
            tokens = self.psg_tokenizer(line)["input_ids"]
            if chunk_size + len(tokens) > paragraph_max_tokens or line.startswith("==") and line.endswith("=="):
                if len(cur_paras) > 0 and cur_section.strip() not in stop_sections:
                    topic = f"{title}: {cur_section}. " if cur_section != '' else f"{title}. "
                    text = topic + ' '.join(cur_paras)

                    passages.append({"title": topic.strip(), "text": text})
                if line.startswith("==") and line.endswith("=="):
                    cur_section = line.strip().strip('=').strip()
                cur_paras.clear()
                chunk_size = 0
            if line.startswith("==") and line.endswith("=="):
                continue
            cur_paras.append(line)
            chunk_size += len(tokens)
        if len(cur_paras) > 0 and cur_section.strip() not in stop_sections:
            topic = f"{title}: {cur_section}. "
            text = topic + ' '.join(cur_paras)

            passages.append({"title": topic.strip(), "text": text})

        return passages

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)

            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)

            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_bm25_passages(self, topic, query, passages, k, cache=True):
        if topic in self.embed_cache and cache:
            bm25 = self.embed_cache[topic]
        else:
            # if self.context_type == "db":
            inputs = [psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages]
            # elif self.context_type == "wikipedia_api":
            #     # assert isinstance(passages, str)
            #     inputs = [[f"{topic}. {p}", ] for p in passages]
            # else:
            #     raise RuntimeError(f"Invalid context_type: {self.context_type}")
            bm25 = BM25Okapi(inputs)
            self.embed_cache[topic] = bm25
            self.add_n_embed += 1
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device,
                                                  show_progress_bar=False)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        query_vectors = self.encoder.encode([retrieval_query],
                                            batch_size=self.batch_size,
                                            device=self.encoder.device,
                                            show_progress_bar=False)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_passages_local_db(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        passages = self.db.get_text_from_title(topic)

        return passages

        # if cache_key not in self.cache:

        # if self.retrieval_type == "bm25":
        #     self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
        # else:
        #     self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
        # assert len(self.cache[cache_key]) in [k, len(passages)]
        # self.add_n += 1

        # return self.cache[cache_key]

    def rerank_passages(self, cache_key, topic, retrieval_query, passages, k, cache=True):
        if self.retrieval_type == "bm25":
            self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k, cache=cache)
        else:
            self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
        assert len(self.cache[cache_key]) in [k, len(passages)]
        self.add_n += 1

        return self.cache[cache_key]

    def get_passages(self, topic, question, k):
        # cache_key = topic + "#" + retrieval_query

        if self.context_type == "db":
            passages_fn = self.get_passages_local_db
            # return self.get_passages_local_db(topic, question, k)
        elif self.context_type == "wikipedia_api":
            passages_fn = self.get_passages_wikipedia_api
            # return self.get_passages_wikipedia_api(topic, question, k)
        else:
            raise RuntimeError(f"Invalid context retrieval: {self.context_type}")
        if isinstance(topic, str):
            retrieval_query = topic + " " + question.strip()
            cache_key = topic + "#" + retrieval_query

            passages = passages_fn(topic, question, k)
            return self.rerank_passages(cache_key=cache_key,
                                        topic=topic,
                                        retrieval_query=retrieval_query,
                                        passages=passages,
                                        k=k)
            # if self.retrieval_type == "bm25":
            #     self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
            # else:
            #     self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            # assert len(self.cache[cache_key]) in [k, len(passages)]
            # self.add_n += 1
            # return self.cache[cache_key]
        elif isinstance(topic, list):
            passages = []
            retrieval_query = question.strip()
            cache_key = "#" + retrieval_query
            for t in topic:
                psgs = passages_fn(t, question, k)
                passages.extend(psgs)

            return self.rerank_passages(cache_key=cache_key,
                                        topic=question.strip(),
                                        retrieval_query=retrieval_query,
                                        passages=passages,
                                        k=k,
                                        cache=False)
        else:
            raise RuntimeError(f"Invalid topic data type {type(topic)} for topic {topic}")

    def get_passages_wikipedia_api(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key not in self.cache:
            if topic not in self.cache:
                if self.page_search_mode == "single":
                    doc = retrieve_wikipedia_page(topic=topic)
                    passages = self.wikipedia_page2passages(doc=doc)
                    self.cache[topic] = passages
                if self.page_search_mode == "multi":
                    docs = retrieve_multiple_wikipedia_pages(topic=topic, context_num_pages=self.context_num_pages)
                    passages = []
                    for doc in docs:
                        psgs = self.wikipedia_page2passages(doc=doc)
                        passages.extend(psgs)
                    self.cache[topic] = passages

        passages = self.cache[topic]

        return passages

        #     if self.retrieval_type == "bm25":
        #         self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
        #     else:
        #         self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
        #     assert len(self.cache[cache_key]) in [k, len(passages)]
        #     self.add_n += 1
        # return self.cache[cache_key]
