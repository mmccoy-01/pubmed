#!/usr/bin/env python3
"""
Daily PubMed digest for newly indexed LLM-related papers.

The script:
1. Searches PubMed for recent LLM-related additions.
2. Pulls metadata, abstract text, and PMC full text when available.
3. Uses the OpenAI Responses API to score papers for impact and interestingness.
4. Writes a Markdown digest and a machine-readable JSON export.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

from openai import OpenAI

T = TypeVar("T")


TOPIC_PRESETS: dict[str, str] = {
    "llm": textwrap.dedent(
        """
        (
          "large language model"[Title/Abstract]
          OR "large language models"[Title/Abstract]
          OR LLM[Title/Abstract]
          OR GPT-4[Title/Abstract]
          OR GPT-4o[Title/Abstract]
          OR GPT-5[Title/Abstract]
          OR "foundation model"[Title/Abstract]
          OR "foundation models"[Title/Abstract]
          OR "generative AI"[Title/Abstract]
          OR "generative artificial intelligence"[Title/Abstract]
          OR "retrieval augmented generation"[Title/Abstract]
          OR RAG[Title/Abstract]
          OR "transformer model"[Title/Abstract]
          OR "transformer models"[Title/Abstract]
        )
        """
    ).strip(),
    "bioinformatics": textwrap.dedent(
        """
        (
          bioinformatics[Title/Abstract]
          OR genomics[Title/Abstract]
          OR proteomics[Title/Abstract]
          OR transcriptomics[Title/Abstract]
          OR "single-cell"[Title/Abstract]
          OR "computational biology"[Title/Abstract]
          OR "biological sequence"[Title/Abstract]
        )
        AND
        (
          "artificial intelligence"[Title/Abstract]
          OR "machine learning"[Title/Abstract]
          OR "deep learning"[Title/Abstract]
          OR "foundation model"[Title/Abstract]
          OR "large language model"[Title/Abstract]
        )
        """
    ).strip(),
    "neuroscience": textwrap.dedent(
        """
        (
          neuroscience[Title/Abstract]
          OR neuroimaging[Title/Abstract]
          OR MRI[Title/Abstract]
          OR EEG[Title/Abstract]
          OR fMRI[Title/Abstract]
          OR brain[Title/Abstract]
          OR neuron*[Title/Abstract]
        )
        AND
        (
          "artificial intelligence"[Title/Abstract]
          OR "machine learning"[Title/Abstract]
          OR "deep learning"[Title/Abstract]
          OR "foundation model"[Title/Abstract]
          OR "large language model"[Title/Abstract]
        )
        """
    ).strip(),
    "medical-ai": textwrap.dedent(
        """
        (
          clinic*[Title/Abstract]
          OR hospital[Title/Abstract]
          OR patient*[Title/Abstract]
          OR diagnosis[Title/Abstract]
          OR radiology[Title/Abstract]
          OR pathology[Title/Abstract]
          OR surgery[Title/Abstract]
        )
        AND
        (
          "artificial intelligence"[Title/Abstract]
          OR "machine learning"[Title/Abstract]
          OR "deep learning"[Title/Abstract]
          OR "foundation model"[Title/Abstract]
          OR "large language model"[Title/Abstract]
          OR LLM[Title/Abstract]
        )
        """
    ).strip(),
    "nlp": textwrap.dedent(
        """
        (
          "natural language processing"[Title/Abstract]
          OR NLP[Title/Abstract]
          OR "language model"[Title/Abstract]
          OR "large language model"[Title/Abstract]
          OR LLM[Title/Abstract]
          OR "machine translation"[Title/Abstract]
          OR summarization[Title/Abstract]
          OR retrieval[Title/Abstract]
        )
        """
    ).strip(),
}

DEFAULT_QUERY = TOPIC_PRESETS["llm"]

DEFAULT_MODEL = os.getenv("OPENAI_MODEL")
DEFAULT_FINAL_MODEL = os.getenv("OPENAI_FINAL_MODEL")
DEFAULT_DAYS_BACK = int(os.getenv("PUBMED_DAYS_BACK", "3"))
MEDLINE_STAGE_TARGET = 50
BIORXIV_STAGE_TARGET = 80
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DB_PATH = DATA_DIR / "pubmed_digest.sqlite3"
REQUEST_TIMEOUT = 60
MAX_HTTP_RETRIES = 4
DEFAULT_CANDIDATE_POOL_SIZE = 50
DEFAULT_SOURCES = tuple(
    source.strip().lower()
    for source in os.getenv("PUBMED_SOURCES", "pubmed,biorxiv,arxiv").split(",")
    if source.strip()
)
VALID_SOURCES = {"pubmed", "biorxiv", "arxiv"}
ARXIV_REQUEST_TIMEOUT = int(os.getenv("ARXIV_REQUEST_TIMEOUT", "15"))
DEFAULT_SCORING_RULES_PATH = ROOT / "config" / "scoring_rules.json"
DEFAULT_SCORING_MODEL_PREFERENCES = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4"]
DEFAULT_FINAL_MODEL_PREFERENCES = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


@dataclass
class Paper:
    paper_id: str
    source_db: str
    seen_key: str
    title: str
    authors: list[str]
    journal: str
    pubdate: str
    doi: str | None
    pmcid: str | None
    abstract: str
    full_text: str
    source: str
    entry_url: str
    link_label: str
    pmc_url: str | None


def normalize_journal_name(value: str) -> str:
    return collapse_whitespace(value).casefold()


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "pubmed-digest/0.1"})
    with open_with_retry(request) as response:
        return json.loads(response.read().decode("utf-8"))


def http_get_text(url: str, params: dict[str, Any]) -> str:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "pubmed-digest/0.1"})
    with open_with_retry(request) as response:
        return response.read().decode("utf-8", errors="replace")


def http_get_text_absolute(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "pubmed-digest/0.1"})
    with open_with_retry(request) as response:
        return response.read().decode("utf-8", errors="replace")


def http_get_text_absolute_with_timeout(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "pubmed-digest/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def open_with_retry(request: urllib.request.Request):
    for attempt in range(MAX_HTTP_RETRIES):
        try:
            return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == MAX_HTTP_RETRIES - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * (2**attempt)
            time.sleep(delay)
        except urllib.error.URLError:
            if attempt == MAX_HTTP_RETRIES - 1:
                raise
            time.sleep(1.0 * (2**attempt))
    raise RuntimeError("unreachable retry state")


def ncbi_params() -> dict[str, str]:
    params: dict[str, str] = {"retmode": "json"}
    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("NCBI_EMAIL")
    tool_name = os.getenv("NCBI_TOOL", "pubmed-digest")
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    if tool_name:
        params["tool"] = tool_name
    return params


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def daily_output_dir(now: dt.datetime | None = None) -> Path:
    current = now or dt.datetime.now()
    path = OUTPUT_DIR / current.strftime("%Y-%m-%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def list_available_openai_models(api_key: str | None) -> set[str]:
    if not api_key:
        return set()
    client = OpenAI(api_key=api_key)
    try:
        response = client.models.list()
    except Exception:
        return set()
    return {model.id for model in getattr(response, "data", []) if getattr(model, "id", None)}


def choose_preferred_model(available: set[str], preferred: list[str], fallback: str) -> str:
    for candidate in preferred:
        if candidate in available:
            return candidate
    return fallback


def resolve_model_selection(
    api_key: str | None,
    scoring_model: str | None,
    final_model: str | None,
) -> tuple[str, str]:
    scoring_override = scoring_model or os.getenv("OPENAI_MODEL")
    final_override = final_model or os.getenv("OPENAI_FINAL_MODEL")
    if scoring_override and final_override:
        return scoring_override, final_override

    available_models = list_available_openai_models(api_key)
    resolved_final = final_override or choose_preferred_model(
        available_models,
        DEFAULT_FINAL_MODEL_PREFERENCES,
        "gpt-5.4",
    )
    resolved_scoring = scoring_override or choose_preferred_model(
        available_models,
        DEFAULT_SCORING_MODEL_PREFERENCES,
        "gpt-5.4-mini",
    )
    return resolved_scoring, resolved_final


def available_topics() -> list[str]:
    return sorted(TOPIC_PRESETS)


def normalize_sources(raw_sources: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if raw_sources is None:
        sources = set(DEFAULT_SOURCES)
    elif isinstance(raw_sources, str):
        sources = {source.strip().lower() for source in raw_sources.split(",") if source.strip()}
    else:
        sources = {source.strip().lower() for source in raw_sources if source.strip()}

    unknown = sources - VALID_SOURCES
    if unknown:
        raise ValueError(
            f"Unknown source(s): {', '.join(sorted(unknown))}. "
            f"Valid sources: {', '.join(sorted(VALID_SOURCES))}"
        )
    if not sources:
        raise ValueError("At least one source must be enabled.")
    return sources


def resolve_query(topic: str | None, query: str | None, topic_file: Path | None) -> tuple[str, str]:
    if query:
        return query, "custom-query"
    if topic_file:
        return topic_file.read_text(encoding="utf-8").strip(), f"file:{display_path(topic_file) or topic_file.name}"
    chosen_topic = topic or os.getenv("PUBMED_TOPIC", "llm")
    if chosen_topic not in TOPIC_PRESETS:
        raise ValueError(f"Unknown topic '{chosen_topic}'. Available topics: {', '.join(available_topics())}")
    return TOPIC_PRESETS[chosen_topic], chosen_topic


TOPIC_MATCH_RULES: dict[str, dict[str, list[str]]] = {
    "llm": {
        "core_terms": [
            "large language model",
            "large language models",
            "llm",
            "foundation model",
            "foundation models",
            "generative ai",
            "gpt-4",
            "gpt-4o",
            "gpt-5",
            "retrieval augmented generation",
            "rag",
            "transformer model",
            "transformer models",
        ]
    },
    "bioinformatics": {
        "domain_terms": [
            "bioinformatics",
            "genomics",
            "proteomics",
            "transcriptomics",
            "single-cell",
            "computational biology",
            "biological sequence",
            "gene expression",
            "protein structure",
            "rna-seq",
            "crispr",
        ],
        "method_terms": [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "foundation model",
            "large language model",
            "neural network",
        ],
    },
    "neuroscience": {
        "domain_terms": [
            "neuroscience",
            "neuroimaging",
            "mri",
            "eeg",
            "fmri",
            "brain",
            "neuron",
            "connectome",
            "neural activity",
        ],
        "method_terms": [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "foundation model",
            "large language model",
            "neural network",
        ],
    },
    "medical-ai": {
        "domain_terms": [
            "clinical",
            "hospital",
            "patient",
            "diagnosis",
            "radiology",
            "pathology",
            "surgery",
            "electronic health record",
            "ehr",
        ],
        "method_terms": [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "foundation model",
            "large language model",
            "llm",
        ],
    },
    "nlp": {
        "core_terms": [
            "natural language processing",
            "nlp",
            "language model",
            "large language model",
            "llm",
            "machine translation",
            "summarization",
            "retrieval",
            "question answering",
        ]
    },
}


def extract_query_terms(query: str, max_terms: int = 14) -> list[str]:
    quoted = [collapse_whitespace(match) for match in re.findall(r'"([^"]+)"', query)]
    quoted_word_sets = [{part.casefold() for part in phrase.split()} for phrase in quoted]
    words = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+\.]{1,}", query)
        if token.upper() not in {"AND", "OR", "NOT", "TITLE", "ABSTRACT"}
        and not token.endswith("Abstract")
    ]
    terms: list[str] = []
    for value in quoted + words:
        normalized = value.strip()
        if not normalized:
            continue
        if value in words and any(normalized.casefold() in word_set for word_set in quoted_word_sets):
            continue
        if normalized.casefold() not in {term.casefold() for term in terms}:
            terms.append(normalized)
        if len(terms) >= max_terms:
            break
    return terms


def extract_query_term_groups(query: str, max_groups: int = 6, max_terms_per_group: int = 12) -> list[list[str]]:
    group_matches = re.findall(r"\(([^()]+)\)", query)
    groups: list[list[str]] = []
    for group_text in group_matches[:max_groups]:
        terms = extract_query_terms(group_text, max_terms=max_terms_per_group)
        if terms:
            groups.append(terms)
    return groups


def topic_terms(topic_label: str, query: str) -> list[str]:
    rules = TOPIC_MATCH_RULES.get(topic_label)
    if rules:
        terms = rules.get("core_terms", []) + rules.get("domain_terms", []) + rules.get("method_terms", [])
        ordered: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(term)
        return ordered
    return extract_query_terms(query)


def text_matches_terms(text: str, terms: list[str]) -> bool:
    haystack = text.casefold()
    return any(term.casefold() in haystack for term in terms)


def topic_description(topic_label: str) -> str:
    descriptions = {
        "llm": "large language models, foundation models, generative AI, and closely related language-model research",
        "bioinformatics": "AI methods applied to bioinformatics, genomics, proteomics, transcriptomics, single-cell biology, or computational biology",
        "neuroscience": "AI methods applied to neuroscience, neuroimaging, brain data, or neural systems",
        "medical-ai": "AI methods applied to clinical care, hospital workflows, diagnosis, radiology, pathology, or patient care",
        "nlp": "natural language processing, language modeling, translation, retrieval, and related language technologies",
    }
    return descriptions.get(topic_label, f"research centrally about the topic '{topic_label}'")


def text_matches_topic(text: str, topic_label: str, query: str) -> bool:
    haystack = text.casefold()
    rules = TOPIC_MATCH_RULES.get(topic_label)
    if not rules:
        groups = extract_query_term_groups(query)
        if groups:
            return all(any(term.casefold() in haystack for term in group) for group in groups)
        return text_matches_terms(text, extract_query_terms(query))
    core_terms = [term.casefold() for term in rules.get("core_terms", [])]
    if core_terms:
        return any(term in haystack for term in core_terms)
    domain_terms = [term.casefold() for term in rules.get("domain_terms", [])]
    method_terms = [term.casefold() for term in rules.get("method_terms", [])]
    has_domain = any(term in haystack for term in domain_terms) if domain_terms else True
    has_method = any(term in haystack for term in method_terms) if method_terms else True
    return has_domain and has_method


def topic_match_score(text: str, topic_label: str, query: str) -> int:
    haystack = text.casefold()
    rules = TOPIC_MATCH_RULES.get(topic_label)
    if not rules:
        groups = extract_query_term_groups(query)
        if groups:
            score = 0
            matched_groups = 0
            for group in groups:
                group_hits = sum(1 for term in group if term.casefold() in haystack)
                if group_hits:
                    matched_groups += 1
                    score += 5 + group_hits
            if matched_groups == len(groups):
                score += 10
            return score
        return sum(1 for term in extract_query_terms(query) if term.casefold() in haystack)

    core_terms = [term.casefold() for term in rules.get("core_terms", [])]
    if core_terms:
        hits = sum(1 for term in core_terms if term in haystack)
        return hits * 3

    domain_terms = [term.casefold() for term in rules.get("domain_terms", [])]
    method_terms = [term.casefold() for term in rules.get("method_terms", [])]
    domain_hits = sum(1 for term in domain_terms if term in haystack)
    method_hits = sum(1 for term in method_terms if term in haystack)
    score = domain_hits * 3 + method_hits * 2
    if domain_hits and method_hits:
        score += 10
    return score


def tighten_scored_items(
    items: list[T],
    slots_available: int,
    text_getter: Callable[[T], str],
    topic_label: str,
    query: str,
) -> tuple[list[T], bool]:
    if slots_available <= 0:
        return [], bool(items)
    if len(items) <= slots_available:
        return items, False

    scored_items = []
    for index, item in enumerate(items):
        text = text_getter(item)
        scored_items.append((topic_match_score(text, topic_label, query), index, item))

    scored_items.sort(key=lambda row: (-row[0], row[1]))
    tightened = [item for _, _, item in scored_items[:slots_available]]
    return tightened, True


def load_journal_whitelist(path: Path) -> set[str]:
    journals = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        journals.add(normalize_journal_name(line))
    return journals


def load_journal_whitelist_entries(path: Path) -> list[str]:
    journals: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        journals.append(line)
    return journals


def init_db() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_papers (
            pmid TEXT PRIMARY KEY,
            first_seen_at TEXT NOT NULL,
            last_digest_path TEXT
        )
        """
    )
    conn.commit()
    return conn


def search_pubmed(query: str, days_back: int, retmax: int) -> list[str]:
    params = {
        **ncbi_params(),
        "db": "pubmed",
        "term": query,
        "sort": "pub date",
        "reldate": str(days_back),
        "datetype": "edat",
        "retmax": str(retmax),
    }
    payload = http_get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params)
    return payload.get("esearchresult", {}).get("idlist", [])


def arxiv_clause(terms: list[str]) -> str:
    parts = []
    for term in terms:
        if " " in term or "-" in term or "+" in term:
            parts.append(f'all:"{term}"')
        else:
            parts.append(f"all:{term}")
    return "(" + " OR ".join(parts) + ")"


def build_arxiv_query(topic_label: str, query: str) -> str:
    rules = TOPIC_MATCH_RULES.get(topic_label)
    if rules and rules.get("core_terms"):
        return "(cat:cs.*) AND " + arxiv_clause(rules["core_terms"])
    if rules and rules.get("domain_terms") and rules.get("method_terms"):
        return "(cat:cs.*) AND " + arxiv_clause(rules["domain_terms"]) + " AND " + arxiv_clause(rules["method_terms"])
    return "(cat:cs.*) AND " + arxiv_clause(topic_terms(topic_label, query))


def search_arxiv_cs(days_back: int, retmax: int, topic_label: str, query: str) -> list[dict[str, Any]]:
    params = {
        "search_query": build_arxiv_query(topic_label, query),
        "start": "0",
        "max_results": str(retmax),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    feed = http_get_text_absolute_with_timeout(
        "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params),
        timeout=ARXIV_REQUEST_TIMEOUT,
    )
    root = ET.fromstring(feed)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = collapse_whitespace(entry.findtext("atom:id", default="", namespaces=ns))
        if not entry_id:
            continue
        published_raw = collapse_whitespace(entry.findtext("atom:published", default="", namespaces=ns))
        try:
            published_dt = dt.datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except ValueError:
            published_dt = None
        if published_dt and published_dt < cutoff:
            continue
        links = entry.findall("atom:link", ns)
        pdf_url = None
        for link in links:
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
        authors = [
            collapse_whitespace(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
            if collapse_whitespace(author.findtext("atom:name", default="", namespaces=ns))
        ]
        primary_category = entry.find("arxiv:primary_category", ns)
        entries.append(
            {
                "paper_id": entry_id.rsplit("/", 1)[-1],
                "entry_url": entry_id,
                "title": collapse_whitespace(entry.findtext("atom:title", default="", namespaces=ns)),
                "abstract": collapse_whitespace(entry.findtext("atom:summary", default="", namespaces=ns)),
                "authors": authors,
                "pubdate": published_raw[:10] if published_raw else "",
                "journal": primary_category.attrib.get("term", "arXiv cs") if primary_category is not None else "arXiv cs",
                "doi": collapse_whitespace(entry.findtext("arxiv:doi", default="", namespaces=ns)) or None,
                "pdf_url": pdf_url,
            }
        )
    return entries


def fetch_arxiv_entry_map(days_back: int, retmax: int, topic_label: str, query: str) -> dict[str, dict[str, Any]]:
    return {entry["paper_id"]: entry for entry in search_arxiv_cs(days_back, retmax, topic_label, query)}


def search_biorxiv(days_back: int, retmax: int, topic_label: str, query: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    cursor = 0
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days_back)
    interval = f"{start_date.isoformat()}/{end_date.isoformat()}"
    while len(entries) < retmax:
        response = http_get_json(f"https://api.biorxiv.org/details/biorxiv/{interval}/{cursor}/json", {})
        batch = response.get("collection", [])
        if not batch:
            break
        matched_in_batch = 0
        for item in batch:
            title = collapse_whitespace(item.get("title", ""))
            abstract = collapse_whitespace(item.get("abstract", ""))
            combined = f"{title}\n{abstract}"
            if not text_matches_topic(combined, topic_label, query):
                continue
            doi = collapse_whitespace(item.get("doi", ""))
            if not doi:
                continue
            version = collapse_whitespace(str(item.get("version", ""))) or "1"
            category = collapse_whitespace(item.get("category", "")) or "bioRxiv"
            authors_raw = item.get("authors", "")
            authors = [collapse_whitespace(name) for name in authors_raw.split(";") if collapse_whitespace(name)]
            entries.append(
                {
                    "paper_id": doi,
                    "entry_url": f"https://www.biorxiv.org/content/{doi}v{version}",
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "pubdate": collapse_whitespace(item.get("date", "")),
                    "journal": f"bioRxiv: {category}",
                    "doi": doi,
                    "pdf_url": f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf",
                }
            )
            matched_in_batch += 1
            if len(entries) >= retmax:
                break
        if len(batch) < 100:
            break
        cursor += len(batch)
        if matched_in_batch == 0 and cursor >= 300:
            break
    return entries


def fetch_biorxiv_entry_map(days_back: int, retmax: int, topic_label: str, query: str) -> dict[str, dict[str, Any]]:
    return {entry["paper_id"]: entry for entry in search_biorxiv(days_back, retmax, topic_label, query)}


def build_whitelist_journal_query(journal_names: list[str]) -> str:
    terms = [f'"{name}"[Journal]' for name in journal_names]
    return "(" + " OR ".join(terms) + ")"


def build_candidate_pmids(
    conn: sqlite3.Connection,
    query: str,
    topic_label: str,
    days_back: int,
    candidate_pool_size: int,
    journal_whitelist_path: Path | None,
    sources: set[str],
) -> tuple[list[str], dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    collected: list[str] = []
    seen_pmids = {row[0] for row in conn.execute("SELECT pmid FROM seen_papers")}

    def add_pubmed_stage(stage_name: str, stage_query: str, stage_retmax: int, stage_target: int) -> None:
        nonlocal collected
        slots_available = max(0, stage_target - len(collected))
        if slots_available <= 0:
            return
        pmids = search_pubmed(stage_query, days_back, stage_retmax)
        candidate_pmids = []
        for pmid in pmids:
            seen_key = f"pubmed:{pmid}"
            if seen_key in seen_pmids or seen_key in collected:
                continue
            candidate_pmids.append(pmid)

        summaries = fetch_summaries(candidate_pmids) if len(candidate_pmids) > slots_available else {}
        tightened_pmids, tightened = tighten_scored_items(
            candidate_pmids,
            slots_available,
            text_getter=lambda pmid: (
                collapse_whitespace(
                    f"{summaries.get(pmid, {}).get('title', '')}\n"
                    f"{summaries.get(pmid, {}).get('fulljournalname', summaries.get(pmid, {}).get('source', ''))}"
                )
                if summaries
                else pmid
            ),
            topic_label=topic_label,
            query=query,
        )
        for pmid in tightened_pmids:
            collected.append(f"pubmed:{pmid}")
        stages.append(
            {
                "stage": stage_name,
                "source": "pubmed",
                "retmax": stage_retmax,
                "query": stage_query,
                "returned": len(pmids),
                "slots_available": slots_available,
                "added": len(tightened_pmids),
                "tightened": tightened,
                "pool_size_after_stage": len(collected),
            }
        )

    def add_biorxiv_stage(stage_name: str, stage_retmax: int, stage_target: int) -> None:
        nonlocal collected
        slots_available = max(0, stage_target - len(collected))
        if slots_available <= 0:
            return
        try:
            entries = search_biorxiv(days_back, stage_retmax, topic_label, query)
            stage_error = None
        except Exception as exc:  # noqa: BLE001
            entries = []
            stage_error = str(exc)
        candidate_entries = []
        for entry in entries:
            seen_key = f"biorxiv:{entry['paper_id']}"
            if seen_key in seen_pmids or seen_key in collected:
                continue
            candidate_entries.append(entry)
        tightened_entries, tightened = tighten_scored_items(
            candidate_entries,
            slots_available,
            text_getter=lambda entry: f"{entry.get('title', '')}\n{entry.get('abstract', '')}",
            topic_label=topic_label,
            query=query,
        )
        for entry in tightened_entries:
            collected.append(f"biorxiv:{entry['paper_id']}")
        stages.append(
            {
                "stage": stage_name,
                "source": "biorxiv",
                "retmax": stage_retmax,
                "query": ", ".join(topic_terms(topic_label, query)),
                "returned": len(entries),
                "slots_available": slots_available,
                "added": len(tightened_entries),
                "tightened": tightened,
                "pool_size_after_stage": len(collected),
                "error": stage_error,
            }
        )

    def add_arxiv_stage(stage_name: str, stage_retmax: int, stage_target: int) -> None:
        nonlocal collected
        slots_available = max(0, stage_target - len(collected))
        if slots_available <= 0:
            return
        try:
            entries = search_arxiv_cs(days_back, stage_retmax, topic_label, query)
            stage_error = None
        except Exception as exc:  # noqa: BLE001
            entries = []
            stage_error = str(exc)
        candidate_entries = []
        for entry in entries:
            seen_key = f"arxiv:{entry['paper_id']}"
            if seen_key in seen_pmids or seen_key in collected:
                continue
            candidate_entries.append(entry)
        tightened_entries, tightened = tighten_scored_items(
            candidate_entries,
            slots_available,
            text_getter=lambda entry: f"{entry.get('title', '')}\n{entry.get('abstract', '')}",
            topic_label=topic_label,
            query=query,
        )
        for entry in tightened_entries:
            collected.append(f"arxiv:{entry['paper_id']}")
        stages.append(
            {
                "stage": stage_name,
                "source": "arxiv",
                "retmax": stage_retmax,
                "query": build_arxiv_query(topic_label, query),
                "returned": len(entries),
                "slots_available": slots_available,
                "added": len(tightened_entries),
                "tightened": tightened,
                "pool_size_after_stage": len(collected),
                "error": stage_error,
            }
        )

    stage_retmax = max(candidate_pool_size * 3, 100)
    pubmed_stage_target = min(candidate_pool_size, MEDLINE_STAGE_TARGET)
    biorxiv_stage_target = min(candidate_pool_size, BIORXIV_STAGE_TARGET)
    if "pubmed" in sources:
        if journal_whitelist_path:
            whitelist_entries = load_journal_whitelist_entries(journal_whitelist_path)
            whitelist_query = build_whitelist_journal_query(whitelist_entries)
            add_pubmed_stage("journal_whitelist", f"({query}) AND {whitelist_query}", stage_retmax, pubmed_stage_target)
            if len(collected) < pubmed_stage_target:
                add_pubmed_stage("medline_fallback", f"({query}) AND MEDLINE[sb]", stage_retmax, pubmed_stage_target)
        else:
            add_pubmed_stage("default", query, stage_retmax, pubmed_stage_target)
    else:
        stages.append({"stage": "pubmed_skipped", "source": "pubmed", "skipped": True, "pool_size_after_stage": len(collected)})

    if "biorxiv" in sources and len(collected) < biorxiv_stage_target:
        add_biorxiv_stage("biorxiv_fallback", stage_retmax, biorxiv_stage_target)
    elif "biorxiv" not in sources:
        stages.append({"stage": "biorxiv_skipped", "source": "biorxiv", "skipped": True, "pool_size_after_stage": len(collected)})

    if "arxiv" in sources and len(collected) < candidate_pool_size:
        add_arxiv_stage("arxiv_cs_fallback", stage_retmax, candidate_pool_size)
    elif "arxiv" not in sources:
        stages.append({"stage": "arxiv_skipped", "source": "arxiv", "skipped": True, "pool_size_after_stage": len(collected)})

    return collected[:candidate_pool_size], {
        "stages": stages,
        "candidate_pool_size": candidate_pool_size,
        "sources": sorted(sources),
        "arxiv_request_timeout_seconds": ARXIV_REQUEST_TIMEOUT,
    }


def fetch_summaries(pmids: list[str]) -> dict[str, dict[str, Any]]:
    if not pmids:
        return {}
    params = {
        **ncbi_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
    }
    payload = http_get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params)
    result = payload.get("result", {})
    return {pmid: result[pmid] for pmid in pmids if pmid in result}


def fetch_pubmed_article_xml(pmid: str) -> ET.Element:
    params = {
        **ncbi_params(),
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }
    text = http_get_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params)
    return ET.fromstring(text)


def fetch_pmc_article_xml(pmcid: str) -> ET.Element:
    params = {
        **ncbi_params(),
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
    }
    text = http_get_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params)
    return ET.fromstring(text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_abstract(pubmed_root: ET.Element) -> str:
    sections: list[str] = []
    for abstract in pubmed_root.findall(".//Abstract"):
        for elem in abstract.findall("./AbstractText"):
            label = elem.attrib.get("Label")
            text = collapse_whitespace("".join(elem.itertext()))
            if not text:
                continue
            sections.append(f"{label}: {text}" if label else text)
    return "\n\n".join(sections)


def parse_pmc_full_text(pmc_root: ET.Element, char_limit: int) -> str:
    body = pmc_root.find(".//body")
    if body is None:
        return ""

    sections: list[str] = []
    for sec in body.findall(".//sec"):
        title_elem = sec.find("./title")
        title = collapse_whitespace("".join(title_elem.itertext())) if title_elem is not None else ""
        paragraphs = []
        for paragraph in sec.findall("./p"):
            text = collapse_whitespace("".join(paragraph.itertext()))
            if text:
                paragraphs.append(text)
        if paragraphs:
            joined = "\n\n".join(paragraphs)
            sections.append(f"{title}\n{joined}" if title else joined)

    if not sections:
        sections = [
            collapse_whitespace("".join(p.itertext()))
            for p in body.findall(".//p")
            if collapse_whitespace("".join(p.itertext()))
        ]

    text = "\n\n".join(sections)
    return text[:char_limit]


def paper_from_summary(pmid: str, summary: dict[str, Any], full_text_limit: int) -> Paper:
    pubmed_root = fetch_pubmed_article_xml(pmid)
    abstract = parse_abstract(pubmed_root)

    article_ids = {item.get("idtype"): item.get("value") for item in summary.get("articleids", []) if item.get("idtype")}
    pmcid = article_ids.get("pmc")
    full_text = ""
    source = "abstract_only"
    pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None

    if pmcid:
        try:
            pmc_root = fetch_pmc_article_xml(pmcid)
            full_text = parse_pmc_full_text(pmc_root, full_text_limit)
            if full_text:
                source = "pmc_full_text"
        except (urllib.error.URLError, ET.ParseError):
            full_text = ""

    authors = [author.get("name", "").strip() for author in summary.get("authors", []) if author.get("name")]
    return Paper(
        paper_id=pmid,
        source_db="pubmed",
        seen_key=f"pubmed:{pmid}",
        title=collapse_whitespace(summary.get("title", "")),
        authors=authors,
        journal=collapse_whitespace(summary.get("fulljournalname", summary.get("source", ""))),
        pubdate=collapse_whitespace(summary.get("pubdate", "")),
        doi=article_ids.get("doi"),
        pmcid=pmcid,
        abstract=abstract,
        full_text=full_text,
        source=source,
        entry_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        link_label="PubMed",
        pmc_url=pmc_url,
    )


def paper_from_arxiv_entry(entry: dict[str, Any]) -> Paper:
    return Paper(
        paper_id=entry["paper_id"],
        source_db="arxiv",
        seen_key=f"arxiv:{entry['paper_id']}",
        title=entry["title"],
        authors=entry["authors"],
        journal=entry["journal"],
        pubdate=entry["pubdate"],
        doi=entry.get("doi"),
        pmcid=None,
        abstract=entry["abstract"],
        full_text="",
        source="abstract_only",
        entry_url=entry["entry_url"],
        link_label="arXiv",
        pmc_url=entry.get("pdf_url"),
    )


def paper_from_biorxiv_entry(entry: dict[str, Any]) -> Paper:
    return Paper(
        paper_id=entry["paper_id"],
        source_db="biorxiv",
        seen_key=f"biorxiv:{entry['paper_id']}",
        title=entry["title"],
        authors=entry["authors"],
        journal=entry["journal"],
        pubdate=entry["pubdate"],
        doi=entry.get("doi"),
        pmcid=None,
        abstract=entry["abstract"],
        full_text="",
        source="abstract_only",
        entry_url=entry["entry_url"],
        link_label="bioRxiv",
        pmc_url=entry.get("pdf_url"),
    )


def fetch_new_papers(
    conn: sqlite3.Connection,
    query: str,
    topic_label: str,
    days_back: int,
    retmax: int,
    full_text_limit: int,
    journal_whitelist: set[str] | None = None,
    journal_whitelist_path: Path | None = None,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    sources: set[str] | None = None,
) -> tuple[list[Paper], dict[str, Any]]:
    resolved_sources = normalize_sources(sources)
    candidate_ids, search_metadata = build_candidate_pmids(
        conn=conn,
        query=query,
        topic_label=topic_label,
        days_back=days_back,
        candidate_pool_size=candidate_pool_size,
        journal_whitelist_path=journal_whitelist_path,
        sources=resolved_sources,
    )
    if not candidate_ids:
        return [], search_metadata

    pubmed_pmids = [item.split(":", 1)[1] for item in candidate_ids if item.startswith("pubmed:")]
    summaries = fetch_summaries(pubmed_pmids)
    biorxiv_entries = (
        fetch_biorxiv_entry_map(days_back, max(candidate_pool_size * 3, 100), topic_label, query)
        if "biorxiv" in resolved_sources
        else {}
    )
    if "arxiv" in resolved_sources and any(item.startswith("arxiv:") for item in candidate_ids):
        try:
            arxiv_entries = fetch_arxiv_entry_map(days_back, max(candidate_pool_size * 3, 100), topic_label, query)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: arXiv fetch failed while hydrating records; continuing without arXiv papers: {exc}", file=sys.stderr)
            arxiv_entries = {}
    else:
        arxiv_entries = {}
    if journal_whitelist:
        filtered_candidates = []
        for candidate_id in candidate_ids:
            if candidate_id.startswith("pubmed:"):
                pmid = candidate_id.split(":", 1)[1]
                summary = summaries.get(pmid)
                if not summary:
                    continue
                journal_name = normalize_journal_name(summary.get("fulljournalname", summary.get("source", "")))
                if journal_name in journal_whitelist:
                    filtered_candidates.append(candidate_id)
            else:
                filtered_candidates.append(candidate_id)
        candidate_ids = filtered_candidates

    papers = []
    for candidate_id in candidate_ids:
        if candidate_id.startswith("pubmed:"):
            pmid = candidate_id.split(":", 1)[1]
            summary = summaries.get(pmid)
            if not summary:
                continue
            try:
                papers.append(paper_from_summary(pmid, summary, full_text_limit))
                time.sleep(0.34)
            except (urllib.error.URLError, ET.ParseError) as exc:
                print(f"warning: failed to fetch PMID {pmid}: {exc}", file=sys.stderr)
        else:
            source_db, external_id = candidate_id.split(":", 1)
            if source_db == "biorxiv":
                entry = biorxiv_entries.get(external_id)
                if entry:
                    papers.append(paper_from_biorxiv_entry(entry))
            else:
                entry = arxiv_entries.get(external_id)
                if entry:
                    papers.append(paper_from_arxiv_entry(entry))
    search_metadata["papers_fetched"] = len(papers)
    return papers, search_metadata


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    fragments: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                fragments.append(content["text"])
    return "\n".join(fragments).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("model did not return JSON")
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("unterminated JSON object")


def build_analysis_prompt(paper: Paper) -> str:
    content = paper.full_text or paper.abstract
    if not content:
        content = "No abstract or full text available."
    authors = ", ".join(paper.authors[:12]) if paper.authors else "Unknown"
    return textwrap.dedent(
        f"""
        Evaluate this newly discovered research paper for an LLM reading digest.

        Title: {paper.title}
        Source database: {paper.source_db}
        Identifier: {paper.paper_id}
        Journal: {paper.journal}
        Publication date: {paper.pubdate}
        Authors: {authors}
        DOI: {paper.doi or "N/A"}
        Content source: {paper.source}

        Paper text:
        {content[:120000]}
        """
    ).strip()


def analyze_paper(paper: Paper, client: OpenAI, model: str, topic_label: str) -> dict[str, Any]:
    prompt = build_analysis_prompt(paper)
    instructions = textwrap.dedent(
        """
        You are scoring papers for a daily topic-specific research digest.
        Return exactly one JSON object and no surrounding markdown.

        Required schema:
        {
          "topic_relevance_score": 0-10 number,
          "impact_score": 0-10 number,
          "interestingness_score": 0-10 number,
          "awe_factor": 0-10 number,
          "surprise_factor": 0-10 number,
          "rigor_score": 0-10 number,
          "overall_recommendation_score": 0-100 integer,
          "recommendation_label": "must-read" | "worth-reading" | "skim" | "skip",
          "one_paragraph_summary": string,
          "why_it_matters": [string, string, ...],
          "concerns": [string, string, ...],
          "target_reader": string
        }

        Scoring guidance:
        - The chosen topic is: TOPIC_LABEL.
        - The topic should be interpreted as: TOPIC_DESCRIPTION.
        - Reward papers that are genuinely central to the chosen topic, not just loosely adjacent.
        - Penalize generic AI/ML papers that lack a clear, substantial connection to the chosen topic.
        - Reward novelty, study quality, likely influence, and genuinely useful insights.
        - Use awe_factor for work that feels especially impressive, ambitious, elegant, or field-shifting.
        - Use surprise_factor for unexpected findings, unusual applications, counterintuitive results, or clever combinations.
        - Penalize hype, weak evaluation, vague methods, or only marginal topic relevance.
        - If a paper is off-topic, topic_relevance_score should be low and overall_recommendation_score should be strongly reduced.
        """
    ).replace("TOPIC_LABEL", topic_label).replace("TOPIC_DESCRIPTION", topic_description(topic_label)).strip()

    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        instructions=instructions,
        input=prompt,
    )
    text = getattr(response, "output_text", "") or extract_response_text(response.model_dump())
    parsed = extract_json_object(text)
    parsed["model"] = model
    return parsed



def load_scoring_rules(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    haystack = text.casefold()
    needle = term.casefold()
    if re.fullmatch(r"[A-Za-z0-9_+.-]+", needle):
        return re.search(rf"(?<![A-Za-z0-9_+.-]){re.escape(needle)}(?![A-Za-z0-9_+.-])", haystack) is not None
    return needle in haystack


def matched_terms_in_text(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if contains_term(text, term)]


def fielded_term_score(paper: Paper, terms: list[str], weight: float, field_weights: dict[str, float]) -> tuple[float, list[str]]:
    title = paper.title or ""
    abstract = paper.abstract or ""
    journal = paper.journal or ""
    matched: list[str] = []
    score = 0.0
    for term in terms:
        term_score = 0.0
        term_matched = False
        if contains_term(title, term):
            term_score += weight * float(field_weights.get("title_multiplier", 2.0))
            term_matched = True
        if contains_term(abstract, term):
            term_score += weight * float(field_weights.get("abstract_multiplier", 1.0))
            term_matched = True
        if contains_term(journal, term):
            term_score += weight * float(field_weights.get("journal_multiplier", 0.5))
            term_matched = True
        if term_matched:
            matched.append(term)
            score += term_score
    return score, matched


def local_recommendation_label(score: float) -> str:
    if score >= 80:
        return "core must-read"
    if score >= 60:
        return "highly relevant"
    if score >= 40:
        return "worth-reading"
    if score >= 20:
        return "bridge/peripheral"
    if score >= 0:
        return "low-priority"
    return "likely false positive"


def analyze_paper_locally(paper: Paper, scoring_rules: dict[str, Any]) -> dict[str, Any]:
    field_weights = scoring_rules.get("field_weights", {})
    title_abstract_journal = "\n".join([paper.title or "", paper.abstract or "", paper.journal or ""])

    category_matches: dict[str, list[str]] = {}
    category_scores: dict[str, float] = {}
    evidence: list[str] = []
    tags: list[str] = []
    raw_positive_score = 0.0

    for category in scoring_rules.get("positive_categories", []):
        category_id = category.get("id", "unknown")
        terms = category.get("terms", [])
        score, matched = fielded_term_score(
            paper,
            terms,
            float(category.get("weight", 1)),
            field_weights,
        )
        max_score = category.get("max_score")
        if max_score is not None:
            score = min(score, float(max_score))
        if matched:
            category_matches[category_id] = matched
            category_scores[category_id] = round(score, 2)
            raw_positive_score += score
            label = category.get("label", category_id)
            evidence.append(f"{label}: {', '.join(matched[:8])}")
            tags.append(label)
        else:
            category_matches[category_id] = []
            category_scores[category_id] = 0.0

    penalties: dict[str, float] = {}
    total_penalty = 0.0
    concerns: list[str] = []
    for category in scoring_rules.get("negative_categories", []):
        category_id = category.get("id", "unknown")
        matched = matched_terms_in_text(title_abstract_journal, category.get("terms", []))
        if not matched:
            penalties[category_id] = 0.0
            continue
        softeners = matched_terms_in_text(title_abstract_journal, category.get("softening_terms", []))
        penalty = float(category.get("penalty", 0))
        if softeners:
            penalty *= 0.5
        max_penalty = category.get("max_penalty")
        if max_penalty is not None:
            penalty = max(penalty, float(max_penalty))
        penalties[category_id] = round(penalty, 2)
        total_penalty += penalty
        label = category.get("label", category_id)
        if softeners:
            concerns.append(f"Possible {label.lower()} signal, softened by bridge terms: {', '.join(softeners[:5])}.")
        else:
            concerns.append(f"Possible {label.lower()} signal: {', '.join(matched[:5])}.")

    interaction_bonuses: dict[str, float] = {}
    total_interaction_bonus = 0.0
    for rule in scoring_rules.get("interaction_bonus_rules", []):
        fires = False
        if "requires_any_from_each_category" in rule:
            fires = all(category_matches.get(cat_id) for cat_id in rule.get("requires_any_from_each_category", []))
        elif "requires_terms_any" in rule:
            terms_fire = any(contains_term(title_abstract_journal, term) for term in rule.get("requires_terms_any", []))
            category_fire = True
            if rule.get("requires_category_any"):
                category_fire = any(category_matches.get(cat_id) for cat_id in rule.get("requires_category_any", []))
            fires = terms_fire and category_fire
        if fires:
            bonus = float(rule.get("bonus", 0))
            interaction_bonuses[rule.get("id", "unknown")] = bonus
            total_interaction_bonus += bonus
            evidence.append(f"Interaction bonus: {rule.get('label', rule.get('id', 'unknown'))}")
        else:
            interaction_bonuses[rule.get("id", "unknown")] = 0.0

    source_info = scoring_rules.get("source_modifiers", {}).get(paper.source_db, {})
    source_bonus = float(source_info.get("bonus", 0))

    raw_score = raw_positive_score + total_interaction_bonus + source_bonus + total_penalty
    promotion_flags: list[dict[str, Any]] = []
    promotion_bonus = 0.0
    for rule in scoring_rules.get("preprint_promotion_rules", []):
        if paper.source_db not in set(rule.get("eligible_sources", [])):
            continue
        if raw_score < float(rule.get("minimum_raw_score", 0)):
            continue
        required_terms = rule.get("requires_terms_any", [])
        if required_terms and not any(contains_term(title_abstract_journal, term) for term in required_terms):
            continue
        bonus = float(rule.get("promotion_bonus", 0))
        promotion_bonus += bonus
        flag = {
            "id": rule.get("id"),
            "label": rule.get("label"),
            "promotion_bonus": bonus,
            "promote_to_editorial_review": bool(rule.get("promote_to_editorial_review", False)),
        }
        promotion_flags.append(flag)
        evidence.append(f"Preprint promotion: {rule.get('label', rule.get('id', 'unknown'))}")

    final_score = raw_score + promotion_bonus
    final_score_rounded = int(round(final_score))
    topic_relevance_score = max(0, min(10, round(final_score / 10, 1)))
    label = local_recommendation_label(final_score)

    why_it_matters = evidence[:4] if evidence else ["Matched the retrieval query, but did not strongly match the local ontology."]
    if source_info.get("label"):
        why_it_matters.append(f"Source modifier: {source_info['label']}.")
    if not concerns:
        concerns = ["No major local rule-based concerns flagged."]

    if evidence:
        summary = f"Locally scored against the {scoring_rules.get('ontology_name', 'custom')} ontology. Strongest signals: {'; '.join(evidence[:3])}."
    else:
        summary = "Collected by the query, but the local ontology did not find strong priority signals."

    return {
        "topic_relevance_score": topic_relevance_score,
        "impact_score": "n/a",
        "interestingness_score": "n/a",
        "awe_factor": "n/a",
        "surprise_factor": "n/a",
        "rigor_score": "n/a",
        "overall_recommendation_score": final_score_rounded,
        "recommendation_label": label,
        "one_paragraph_summary": summary,
        "why_it_matters": why_it_matters,
        "concerns": concerns,
        "target_reader": "Michael's Hedgehog/taste-organ literature triage workflow",
        "scoring_method": "local_rules",
        "score_breakdown": {
            "positive_categories": category_scores,
            "penalties": penalties,
            "interaction_bonuses": interaction_bonuses,
            "source_bonus": source_bonus,
            "promotion_bonus": promotion_bonus,
            "raw_positive_score": round(raw_positive_score, 2),
            "total_penalty": round(total_penalty, 2),
            "total_interaction_bonus": round(total_interaction_bonus, 2),
            "final_score": final_score_rounded,
        },
        "matched_terms": category_matches,
        "tags": tags,
        "promotion_flags": promotion_flags,
    }

def analyze_papers(
    papers: list[Paper],
    api_key: str | None,
    model: str,
    topic_label: str,
    scoring_rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results = []
    client = OpenAI(api_key=api_key) if api_key else None
    for paper in papers:
        analysis: dict[str, Any]
        if api_key:
            try:
                assert client is not None
                analysis = analyze_paper(paper, client, model, topic_label)
            except Exception as exc:  # noqa: BLE001
                analysis = {
                    "error": str(exc),
                    "topic_relevance_score": 0,
                    "recommendation_label": "unscored",
                    "overall_recommendation_score": 0,
                    "one_paragraph_summary": "OpenAI scoring failed for this paper.",
                    "why_it_matters": [],
                    "concerns": [str(exc)],
                    "target_reader": "Unknown",
                }
        else:
            if scoring_rules:
                analysis = analyze_paper_locally(paper, scoring_rules)
            else:
                analysis = {
                    "topic_relevance_score": 0,
                    "recommendation_label": "unscored",
                    "overall_recommendation_score": 0,
                    "one_paragraph_summary": "No OPENAI_API_KEY provided and no local scoring rules were loaded, so this paper was collected but not ranked.",
                    "why_it_matters": [],
                    "concerns": ["Set OPENAI_API_KEY or provide --scoring-rules config/scoring_rules.json to enable ranking."],
                    "target_reader": "Unknown",
                }
        results.append({"paper": asdict(paper), "analysis": analysis})
    results.sort(key=lambda item: item["analysis"].get("overall_recommendation_score", 0), reverse=True)
    return results


def rerank_records(records: list[dict[str, Any]], api_key: str | None, model: str, top_k: int, topic_label: str) -> list[dict[str, Any]]:
    if not api_key or not records:
        return records[:top_k]

    compact_records = []
    for record in records:
        paper = record["paper"]
        analysis = record["analysis"]
        compact_records.append(
            {
                "paper_id": paper["paper_id"],
                "source_db": paper["source_db"],
                "title": paper["title"],
                "journal": paper["journal"],
                "pubdate": paper["pubdate"],
                "scores": {
                    "overall": analysis.get("overall_recommendation_score"),
                    "impact": analysis.get("impact_score"),
                    "interestingness": analysis.get("interestingness_score"),
                    "awe": analysis.get("awe_factor"),
                    "surprise": analysis.get("surprise_factor"),
                    "rigor": analysis.get("rigor_score"),
                    "topic_relevance": analysis.get("topic_relevance_score", analysis.get("llm_relevance")),
                },
                "summary": analysis.get("one_paragraph_summary", ""),
                "why_it_matters": analysis.get("why_it_matters", [])[:3],
                "concerns": analysis.get("concerns", [])[:3],
            }
        )

    client = OpenAI(api_key=api_key)
    instructions = textwrap.dedent(
        """
        You are producing the final editorial ranking for a daily topic-specific research digest.
        Return exactly one JSON object and no markdown.

        The chosen topic is: TOPIC_LABEL.
        Prioritize work a human expert should read first, balancing topical centrality, rigor,
        practical impact, conceptual importance, novelty, surprise, and likely lasting value.
        Do not rank generic AI papers highly if they are weakly connected to the chosen topic.

        Required schema:
        {
          "ordered_ids": ["paper_id_1", "paper_id_2", "..."]
        }
        """
    ).replace("TOPIC_LABEL", topic_label).strip()
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        instructions=instructions,
        input=json.dumps({"top_k": top_k, "records": compact_records}, ensure_ascii=True),
    )
    text = getattr(response, "output_text", "") or extract_response_text(response.model_dump())
    payload = extract_json_object(text)
    ordered_ids = payload.get("ordered_ids", [])
    lookup = {record["paper"]["paper_id"]: record for record in records}
    reranked = [lookup[paper_id] for paper_id in ordered_ids if paper_id in lookup]
    seen = {record["paper"]["paper_id"] for record in reranked}
    reranked.extend(record for record in records if record["paper"]["paper_id"] not in seen)
    return reranked[:top_k]


def write_outputs(
    records: list[dict[str, Any]],
    query: str,
    topic_label: str,
    days_back: int,
    journal_whitelist_path: str | None = None,
    search_metadata: dict[str, Any] | None = None,
    scoring_model: str | None = None,
    final_model: str | None = None,
    scored_records: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    now = dt.datetime.now()
    run_dir = daily_output_dir(now)
    markdown_path = run_dir / "digest.md"
    json_path = run_dir / "digest.json"

    display_whitelist_path = display_path(Path(journal_whitelist_path)) if journal_whitelist_path else None

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": now.astimezone(dt.timezone.utc).isoformat(),
                "days_back": days_back,
                "topic": topic_label,
                "query": query,
                "journal_whitelist_path": display_whitelist_path,
                "search_metadata": search_metadata,
                "scoring_model": scoring_model,
                "final_model": final_model,
                "records": records,
                "scored_records": scored_records if scored_records is not None else records,
            },
            handle,
            indent=2,
        )

    lines = [
        f"# PubMed LLM Digest ({dt.datetime.now().strftime('%Y-%m-%d')})",
        "",
        f"- Query window: last {days_back} day(s)",
        f"- Topic: {topic_label}",
        f"- Papers found: {len(records)}",
        f"- Journal whitelist: {display_whitelist_path}" if display_whitelist_path else "- Journal whitelist: none",
        f"- Candidate pool target: {search_metadata.get('candidate_pool_size')}" if search_metadata else "- Candidate pool target: n/a",
        f"- Candidates scored: {search_metadata.get('papers_fetched')}" if search_metadata else "- Candidates scored: n/a",
        f"- Scoring model: {scoring_model}" if scoring_model else "- Scoring model: n/a",
        f"- Final ranking model: {final_model}" if final_model else "- Final ranking model: n/a",
        "",
        "## Recommended Reading",
        "",
    ]

    if not records:
        lines.append("No new papers matched the query window.")
    for index, record in enumerate(records, start=1):
        paper = record["paper"]
        analysis = record["analysis"]
        authors = ", ".join(paper["authors"][:8]) if paper["authors"] else "Unknown authors"
        score = analysis.get("overall_recommendation_score", 0)
        label = analysis.get("recommendation_label", "unscored")
        lines.extend(
            [
                f"### {index}. {paper['title']}",
                "",
                f"- Score: {score} ({label})",
                f"- Subscores: impact {analysis.get('impact_score', 'n/a')}/10, interestingness {analysis.get('interestingness_score', 'n/a')}/10, awe {analysis.get('awe_factor', 'n/a')}/10, surprise {analysis.get('surprise_factor', 'n/a')}/10, rigor {analysis.get('rigor_score', 'n/a')}/10, topic relevance {analysis.get('topic_relevance_score', analysis.get('llm_relevance', 'n/a'))}/10",
                f"- Scoring method: {analysis.get('scoring_method', 'openai' if analysis.get('model') else 'unscored')}",
                f"- Journal: {paper['journal']}",
                f"- Date: {paper['pubdate']}",
                f"- Authors: {authors}",
                f"- Source: {paper['source_db']}",
                f"- {paper['link_label']}: [{paper['paper_id']}]({paper['entry_url']})",
            ]
        )
        if paper.get("pmc_url"):
            full_text_label = "PDF" if paper["source_db"] == "arxiv" else "PMC"
            lines.append(f"- Full text: [{full_text_label}]({paper['pmc_url']})")
        if paper.get("doi"):
            lines.append(f"- DOI: {paper['doi']}")
        breakdown = analysis.get("score_breakdown")
        if breakdown:
            top_categories = sorted(
                breakdown.get("positive_categories", {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:4]
            if top_categories:
                formatted = ", ".join(f"{name}: {value}" for name, value in top_categories if value)
                if formatted:
                    lines.append(f"- Top local signals: {formatted}")
            if breakdown.get("total_penalty"):
                lines.append(f"- Local penalties: {breakdown.get('total_penalty')}")
            if analysis.get("promotion_flags"):
                promo_labels = ", ".join(flag.get("label", flag.get("id", "promotion")) for flag in analysis.get("promotion_flags", []))
                lines.append(f"- Promotion flags: {promo_labels}")
        lines.extend(
            [
                "",
                analysis.get("one_paragraph_summary", "No summary available."),
                "",
                "**Why it matters**",
            ]
        )
        for item in analysis.get("why_it_matters", [])[:4]:
            lines.append(f"- {item}")
        if not analysis.get("why_it_matters"):
            lines.append("- No rationale available.")
        lines.append("")
        lines.append("**Concerns**")
        for item in analysis.get("concerns", [])[:4]:
            lines.append(f"- {item}")
        if not analysis.get("concerns"):
            lines.append("- No major concerns flagged.")
        lines.append("")

    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return markdown_path, json_path


def mark_seen(conn: sqlite3.Connection, records: list[dict[str, Any]], digest_path: Path) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = [
        (record["paper"]["seen_key"], now, str(digest_path))
        for record in records
        if not record["analysis"].get("error")
    ]
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO seen_papers (pmid, first_seen_at, last_digest_path)
        VALUES (?, ?, ?)
        ON CONFLICT(pmid) DO UPDATE SET last_digest_path = excluded.last_digest_path
        """,
        rows,
    )
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days-back",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help="Search the last N days of PubMed/arXiv additions. Defaults to PUBMED_DAYS_BACK or 3.",
    )
    parser.add_argument("--retmax", type=int, default=25, help="Maximum number of PubMed hits to inspect per run.")
    parser.add_argument(
        "--topic",
        default=os.getenv("PUBMED_TOPIC", "llm"),
        choices=available_topics(),
        help="Named topic preset to search.",
    )
    parser.add_argument(
        "--topic-file",
        help="Path to a text file containing a custom PubMed query.",
    )
    parser.add_argument(
        "--query",
        default=os.getenv("PUBMED_QUERY"),
        help="Custom PubMed query string. Overrides --topic and --topic-file.",
    )
    parser.add_argument(
        "--full-text-char-limit",
        type=int,
        default=120000,
        help="Maximum number of full-text characters to send to the LLM.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model for first-pass scoring. If unset, the script queries available models and prefers gpt-5.4-mini.",
    )
    parser.add_argument(
        "--final-model",
        default=DEFAULT_FINAL_MODEL,
        help="OpenAI model for final ranking. If unset, the script queries available models and prefers gpt-5.4.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="Build up to this many candidate papers before ranking them.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated source lanes to use: pubmed,biorxiv,arxiv. Defaults to PUBMED_SOURCES or all three.",
    )
    parser.add_argument(
        "--skip-arxiv",
        action="store_true",
        help="Convenience flag equivalent to removing arxiv from --sources.",
    )
    parser.add_argument(
        "--journal-whitelist",
        help="Path to a newline-delimited journal whitelist file.",
    )
    parser.add_argument(
        "--scoring-rules",
        default=os.getenv("SCORING_RULES_PATH", str(DEFAULT_SCORING_RULES_PATH)),
        help="Path to local JSON scoring rules used when OPENAI_API_KEY is not set.",
    )
    parser.add_argument(
        "--disable-local-scoring",
        action="store_true",
        help="Disable local rule-based scoring when OPENAI_API_KEY is not set.",
    )
    parser.add_argument(
        "--mark-seen-without-scoring",
        action="store_true",
        help="Persist seen PMIDs even when OPENAI_API_KEY is missing.",
    )
    parser.add_argument(
        "--mark-seen-on-error",
        action="store_true",
        help="Persist papers even when OpenAI scoring fails.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    conn = init_db()
    topic_file = Path(args.topic_file).expanduser() if args.topic_file else None
    try:
        resolved_query, topic_label = resolve_query(args.topic, args.query, topic_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    journal_whitelist_path = Path(args.journal_whitelist).expanduser() if args.journal_whitelist else None
    journal_whitelist = load_journal_whitelist(journal_whitelist_path) if journal_whitelist_path else None
    try:
        sources = normalize_sources(args.sources)
        if args.skip_arxiv:
            sources.discard("arxiv")
            if not sources:
                raise ValueError("--skip-arxiv removed the only enabled source. Use --sources pubmed,biorxiv or similar.")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    papers, search_metadata = fetch_new_papers(
        conn=conn,
        query=resolved_query,
        topic_label=topic_label,
        days_back=args.days_back,
        retmax=args.retmax,
        full_text_limit=args.full_text_char_limit,
        journal_whitelist=journal_whitelist,
        journal_whitelist_path=journal_whitelist_path,
        candidate_pool_size=args.candidate_pool_size,
        sources=sources,
    )
    if not papers:
        print("No new matching PubMed papers found.")
        return 0

    api_key = os.getenv("OPENAI_API_KEY")
    scoring_rules = None
    if not api_key and not args.disable_local_scoring:
        scoring_rules_path = Path(args.scoring_rules).expanduser() if args.scoring_rules else DEFAULT_SCORING_RULES_PATH
        if not scoring_rules_path.is_absolute():
            scoring_rules_path = ROOT / scoring_rules_path
        scoring_rules = load_scoring_rules(scoring_rules_path)
        if scoring_rules:
            print(f"Loaded local scoring rules from {scoring_rules_path}")
        else:
            print(f"No local scoring rules found at {scoring_rules_path}; papers will be unscored.")
    scoring_model, final_model = resolve_model_selection(
        api_key=api_key,
        scoring_model=args.model,
        final_model=args.final_model,
    )
    if not api_key and scoring_rules:
        scoring_model = scoring_rules.get("ontology_name", "local-rule-scoring")
        final_model = "local-score-sort"
    records = analyze_papers(
        papers,
        api_key=api_key,
        model=scoring_model,
        topic_label=topic_label,
        scoring_rules=scoring_rules,
    )
    final_records = rerank_records(records, api_key=api_key, model=final_model, top_k=args.retmax, topic_label=topic_label)
    markdown_path, json_path = write_outputs(
        final_records,
        scored_records=records,
        query=resolved_query,
        topic_label=topic_label,
        days_back=args.days_back,
        journal_whitelist_path=str(journal_whitelist_path) if journal_whitelist_path else None,
        search_metadata=search_metadata,
        scoring_model=scoring_model,
        final_model=final_model,
    )

    if args.mark_seen_on_error:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        rows = [(record["paper"]["seen_key"], now, str(markdown_path)) for record in final_records]
        conn.executemany(
            """
            INSERT INTO seen_papers (pmid, first_seen_at, last_digest_path)
            VALUES (?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET last_digest_path = excluded.last_digest_path
            """,
            rows,
        )
        conn.commit()
    elif api_key or args.mark_seen_without_scoring:
        mark_seen(conn, final_records, markdown_path)

    print(f"Wrote Markdown digest to {markdown_path}")
    print(f"Wrote JSON export to {json_path}")
    if not api_key:
        if scoring_rules:
            print("OPENAI_API_KEY is not set; papers were ranked with local scoring rules.")
        else:
            print("OPENAI_API_KEY is not set, so papers were collected but not ranked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
