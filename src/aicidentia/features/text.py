"""Text preprocessing and features: one shared tokenised corpus, TF-IDF, and cached sentence embeddings.

Every topic model (LDA, NMF, BERTopic) is fitted on the *same* tokenised corpus so that their
coherence scores are comparable: lower-case, alphabetic tokens of 3+ letters, standard and domain
stopwords removed, WordNet lemmatisation, then a document-frequency filter.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from aicidentia import cache
from aicidentia.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z]{3,}")
NLTK_DIR = PROJECT_ROOT / "nltk_data"  # git-ignored; WordNet is downloaded here on first use


@dataclass
class Corpus:
    """The shared tokenised corpus.

    `tokens[i]` belongs to `incident_ids[i]`; `docs[i]` is the same tokens joined by spaces (used as
    input to BERTopic's c-TF-IDF); `texts[i]` is the untouched text (used for sentence embeddings).
    """

    incident_ids: list[int]
    texts: list[str]
    tokens: list[list[str]]
    docs: list[str]
    dictionary: Any  # gensim.corpora.Dictionary
    bow: list[list[tuple[int, int]]]
    vocabulary: list[str]
    short_docs: list[int]  # row indices with fewer than `min_tokens` tokens

    @property
    def fingerprint(self) -> str:
        """Hash of the texts and vocabulary: the cache key for anything fitted on this corpus."""
        return cache.hash_parts(cache.hash_texts(self.texts), self.vocabulary)


def topic_settings() -> dict[str, Any]:
    """The `topics:` block of config.yaml."""
    return load_config()["topics"]


@lru_cache(maxsize=1)
def _lemmatiser() -> Any:
    """WordNet lemmatiser; downloads the WordNet corpus into ./nltk_data on first use."""
    import nltk
    from nltk.stem import WordNetLemmatizer

    NLTK_DIR.mkdir(exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        logger.info("Downloading the NLTK WordNet corpus to %s", NLTK_DIR)
        nltk.download("wordnet", download_dir=str(NLTK_DIR), quiet=True)
    return WordNetLemmatizer()


def stopword_set(extra: Sequence[str] | None = None) -> frozenset[str]:
    """scikit-learn's English stopwords plus the domain stopwords from the config."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    domain = topic_settings()["domain_stopwords"] if extra is None else extra
    return frozenset(ENGLISH_STOP_WORDS) | {w.lower() for w in domain}


def lemmatise_word(word: str) -> str:
    """Noun then verb WordNet lemma ('crashes' -> 'crash', 'arrested' -> 'arrest'); keeps 3+ letter results."""
    lemma = _lemmatiser()
    out = lemma.lemmatize(lemma.lemmatize(word, "n"), "v")
    return out if len(out) >= 3 else word


def tokenize(texts: Sequence[str], stopwords: frozenset[str] | None = None) -> list[list[str]]:
    """Lower-case, alphabetic 3+ letter tokens, stopwords removed (before and after lemmatising)."""
    stop = stopwords if stopwords is not None else stopword_set()
    out = []
    for text in texts:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in stop]
        lemmas = [lemmatise_word(w) for w in words]
        out.append([w for w in lemmas if w not in stop])
    return out


def preprocess_text(texts: Sequence[str]) -> list[str]:
    """Tokenise and re-join with spaces (the text BERTopic's c-TF-IDF vectoriser sees)."""
    return [" ".join(t) for t in tokenize(texts)]


def build_bow_corpus(tokens: Sequence[Sequence[str]]) -> tuple[Any, list[list[tuple[int, int]]]]:
    """gensim dictionary and bag-of-words corpus from tokenised texts."""
    from gensim.corpora import Dictionary

    dictionary = Dictionary(list(map(list, tokens)))
    return dictionary, [dictionary.doc2bow(list(t)) for t in tokens]


def build_corpus(incident_ids: Sequence[int], texts: Sequence[str]) -> Corpus:
    """Tokenise, filter the vocabulary by document frequency, and return the shared corpus."""
    from gensim.corpora import Dictionary

    cfg = topic_settings()
    raw = tokenize(texts)
    dictionary = Dictionary(raw)
    dictionary.filter_extremes(no_below=cfg["min_df"], no_above=cfg["max_df"], keep_n=None)
    dictionary.compactify()
    keep = set(dictionary.token2id)
    tokens = [[w for w in doc if w in keep] for doc in raw]
    bow = [dictionary.doc2bow(doc) for doc in tokens]
    vocabulary = [dictionary[i] for i in range(len(dictionary))]
    short = [i for i, doc in enumerate(tokens) if len(doc) < cfg["min_tokens"]]
    logger.info(
        "Corpus: %d docs, %d-word vocabulary, %d short docs",
        len(tokens),
        len(vocabulary),
        len(short),
    )
    return Corpus(
        incident_ids=[int(i) for i in incident_ids],
        texts=list(texts),
        tokens=tokens,
        docs=[" ".join(t) for t in tokens],
        dictionary=dictionary,
        bow=bow,
        vocabulary=vocabulary,
        short_docs=short,
    )


def build_tfidf(tokens: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> tuple[Any, Any]:
    """TF-IDF matrix over pre-tokenised documents with a fixed vocabulary; returns (matrix, vectoriser)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        analyzer=lambda doc: doc, vocabulary=list(vocabulary), sublinear_tf=True
    )
    return vectoriser.fit_transform(list(map(list, tokens))), vectoriser


def embed_sentences(
    texts: Sequence[str], model_name: str, batch_size: int = 32, device: str = "cpu"
) -> np.ndarray:
    """L2-normalised sentence-transformers embeddings, cached in data/interim/.

    Runs on CPU by default for reproducibility; the cache key covers the model name and the texts.
    """

    def compute() -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
        return model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

    return cache.cached(
        "embeddings", (model_name, cache.hash_texts(texts), "normalised"), compute, version="1"
    )
