"""Independent, standard-library-only weekly chemosciences retrieval and ontology ranking.

No AI service, credentials, journal whitelist, or notifications. Existing PhD files
and caches are never read or written. Scores describe topical signal, not rigor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "chemosciences"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def matches(text, alias):
    # Word boundaries avoid gene-name substring matches (e.g. RET in secreted).
    expression = re.escape(alias).replace(r"\*", r"[\w-]*").replace(r"\ ", r"[\s-]+")
    return re.search(r"(?<!\w)" + expression + r"(?!\w)", text, re.I) is not None


def score_paper(paper, config):
    title, abstract = paper.get("title", ""), paper.get("abstract", "")
    hits = []
    for concept in config["concepts"]:
        title_hits = [a for a in concept["terms"] if matches(title, a)]
        abstract_hits = [a for a in concept["terms"] if matches(abstract, a)]
        if title_hits or abstract_hits:
            points = min(concept["cap"], concept["weight"] * (2 if title_hits else 1))
            hits.append(dict(id=concept["id"], label=concept["label"], kind=concept["kind"],
                             points=points, terms=sorted(set(title_hits + abstract_hits)),
                             title_match=bool(title_hits), linkable=concept.get("linkable", False)))
    domains = [h for h in hits if h["kind"] == "domain"]
    biology = [h for h in hits if h["kind"] in {"biology", "mechanism"}]
    hit_ids = {h["id"] for h in hits}
    title_domain = any(h["title_match"] for h in domains)
    domain_alias_count = sum(len(h["terms"]) for h in domains)
    eligible = bool(domains and biology and (title_domain or (domain_alias_count >= 2 and any(h['title_match'] for h in biology))))
    exclusions = []
    if any(t in paper.get("types", []) for t in ["Published Erratum", "Retraction of Publication", "Retracted Publication", "Comment"]):
        eligible = False
        exclusions.append("Correction, comment, or retracted publication; retained for context, not weekly selection.")
    if re.search(r"\b(corrigendum|correction to|authors.? reply|editorial:)\b", title, re.I):
        eligible = False
        exclusions.append("Correction or editorial record.")
    sensor = any(matches(title, t) for t in config["nonbiological_sensor_terms"])
    biological_validation = bool(hit_ids & {"receptor_pharmacology", "sensory_transduction", "sensory_circuits", "cell_identity"})
    if sensor and not biological_validation:
        eligible = False
        exclusions.append("Engineered sensing without a matched biological sensing mechanism.")
    if not domains or not biology:
        exclusions.append("No combination of chemosensory/adjacent domain and biological question matched.")
    elif not eligible and not exclusions:
        exclusions.append("Sensory terminology is weak or incidental in title/abstract.")
    group_scores = {}
    for kind, cap in config["group_caps"].items():
        points = [h["points"] for h in hits if h["kind"] == kind]
        group_scores[kind] = min(cap, max(points, default=0) if kind == 'domain' else sum(points))
    bonuses = []
    for rule in config["interactions"]:
        if all(bool(hit_ids.intersection(group)) for group in rule["requires_any_from_each"]):
            bonuses.append(dict(id=rule["id"], label=rule["label"], points=rule["points"]))
    bonus = min(config["interaction_cap"], sum(b["points"] for b in bonuses))
    raw = sum(group_scores.values()) + bonus
    score = min(100, round(raw)) if eligible else min(20, round(raw))
    prominent = sorted(biology, key=lambda h: (-h["points"], h["id"]))[:3]
    why = "Matched " + ", ".join(h["label"].lower() for h in prominent) + "." if prominent else "Limited ontology signal."
    if bonuses:
        why += " Combination: " + bonuses[0]["label"].lower() + "."
    return dict(score=score, eligible=eligible, concepts=hits, breakdown=group_scores,
                bonuses=bonuses, bonus_total=bonus, why=why, exclusions=exclusions,
                assessment="Title/abstract ontology match; scientific rigor and impact are not assessed.")


def rank_records(papers, config):
    records = [dict(paper=p, analysis=score_paper(p, config)) for p in deduplicate(papers)]
    records.sort(key=lambda r: (-r["analysis"]["score"], r["paper"]["pmid"]))
    eligible = [r for r in records if r["analysis"]["eligible"] and r["analysis"]["score"] >= config["selection"]["minimum_score"]]
    chosen = []
    remaining = eligible.copy()
    for _ in range(min(config["selection"]["digest_size"], len(remaining))):
        reviews_chosen = sum('Review' in r['paper'].get('types', []) for r in chosen)
        options = [r for r in remaining if 'Review' not in r['paper'].get('types', []) or reviews_chosen < config['selection'].get('max_reviews_in_digest', 1)]
        if not options:
            break
        def utility(r):
            concepts = {h["id"] for h in r["analysis"]["concepts"] if h["linkable"]}
            redundancy = max((len(concepts & s) / max(1, len(concepts | s)) for s in [
                {h["id"] for h in c["analysis"]["concepts"] if h["linkable"]} for c in chosen]), default=0)
            return r["analysis"]["score"] - config["selection"]["diversity_penalty"] * redundancy
        best = max(options, key=utility)
        chosen.append(best)
        remaining.remove(best)
    reading = chosen + [r for r in eligible if r not in chosen][:max(0, config["selection"]["reading_list_size"] - len(chosen))]
    for rank, r in enumerate(records, 1):
        r["rank"] = rank
        r["editor_pick"] = r in chosen
        r["reading_list"] = r in reading
        r["editor_order"] = chosen.index(r) + 1 if r in chosen else None
    return records


def deduplicate(papers):
    """Exact PMID/DOI duplicates only; fuzzy titles must not merge distinct studies."""
    seen = set()
    result = []
    for p in papers:
        key = p.get("doi", "").lower().removeprefix("https://doi.org/").strip() or "pmid:" + p["pmid"]
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def request(endpoint, params):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" + endpoint + ".fcgi"
    # POST avoids long-query URL limits. No API key required.
    req = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(), headers={"User-Agent": "chemosciences-digest/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
            time.sleep(.4)  # NCBI's unauthenticated request budget is shared by all modules.
            return body
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def xml_text(element):
    return "".join(element.itertext()) if element is not None else ""


def parse_articles(raw):
    papers = []
    for rec in ET.fromstring(raw).findall("PubmedArticle"):
        article = rec.find("./MedlineCitation/Article")
        pmid = xml_text(rec.find("./MedlineCitation/PMID"))
        identifiers = {e.get("IdType"): xml_text(e) for e in rec.findall("./PubmedData/ArticleIdList/ArticleId")}
        types = [xml_text(x) for x in article.findall("./PublicationTypeList/PublicationType")]
        journal = xml_text(article.find("./Journal/Title"))
        date = article.find("./ArticleDate")
        if date is None:
            date = article.find("./Journal/JournalIssue/PubDate")
        pubdate = "-".join(xml_text(date.find(k)) for k in ["Year", "Month", "Day"] if date is not None and date.find(k) is not None) or xml_text(date)
        authors = [xml_text(a.find("CollectiveName")) or " ".join(filter(None, [xml_text(a.find("LastName")), xml_text(a.find("Initials"))])) for a in article.findall("./AuthorList/Author")]
        related = [dict(type=x.get("RefType"), pmid=xml_text(x.find("PMID"))) for x in rec.findall("./MedlineCitation/CommentsCorrectionsList/CommentsCorrections") if x.find("PMID") is not None]
        papers.append(dict(pmid=pmid, title=xml_text(article.find("ArticleTitle")), abstract="\n\n".join(
            (a.get("Label", "") + ": " if a.get("Label") else "") + xml_text(a) for a in article.findall("./Abstract/AbstractText")),
            authors=authors, journal=journal, pubdate=pubdate, doi=identifiers.get("doi", ""), pmcid=identifiers.get("pmc", ""),
            types=types, preprint="Preprint" in types or bool(re.search(r"biorxiv|medrxiv|research square|arxiv", journal, re.I)),
            url="https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/", related=related))
    return papers


def retrieve(config, start, end):
    ids, metadata = set(), []
    for name, query in config["queries"].items():
        term = f'({query}) AND ("{start}"[EDAT] : "{end}"[EDAT])'
        found = []
        warnings = []
        for offset in range(0, 10000, 500):
            result = json.loads(request("esearch", dict(db="pubmed", term=term, retmode="json", retmax=500, retstart=offset)))["esearchresult"]
            count = int(result["count"])
            if count >= 10000 or result.get("errorlist"):
                raise RuntimeError(f"Query {name} is too broad or invalid; refusing a silently truncated digest.")
            if result.get("warninglist", {}).get("quotedphrasesnotfound"):
                raise RuntimeError(f"PubMed phrase warning in {name}: {result['warninglist']}")
            warnings = result.get("warninglist", {})
            found.extend(result["idlist"])
            if len(found) >= count:
                break
        ids.update(found)
        metadata.append(dict(module=name, count=count, query=term, warnings=warnings))
    papers = []
    ordered = sorted(ids)
    for offset in range(0, len(ordered), 100):
        papers.extend(parse_articles(request("efetch", dict(db="pubmed", id=",".join(ordered[offset:offset+100]), retmode="xml"))))
    return papers, dict(queries=metadata, retrieved_ids=len(ids), article_records=len(papers), source="PubMed (including indexed preprints)", date_field="EDAT")


def rebuild_catalog(output):
    runs, nodes = [], {}
    for path in sorted(output.glob("*/digest.json"), reverse=True):
        d = read_json(path)
        runs.append(dict(id=path.parent.name, start=d["window"]["start"], end=d["window"]["end"], count=sum(r["reading_list"] for r in d["records"]), candidates=len(d["records"]), eligible=sum(r["analysis"]["eligible"] for r in d["records"]), path=path.parent.name + "/digest.json"))
        for r in d["records"]:
            if not r["analysis"]["eligible"]:
                continue
            p = r["paper"]
            node = nodes.setdefault(p["pmid"], dict(id=p["pmid"], title=p["title"], authors=p.get("authors", []), url=p["url"], doi=p.get("doi", ""), related=p.get("related", []), journal=p.get("journal", ""), preprint=p.get("preprint", False), score=r["analysis"]["score"], concepts=r["analysis"]["concepts"], why=r["analysis"]["why"], weeks=[], selected_weeks=[], editor_weeks=[]))
            node["weeks"].append(path.parent.name)
            if r["reading_list"]:
                node["selected_weeks"].append(path.parent.name)
            if r["editor_pick"]:
                node["editor_weeks"].append(path.parent.name)
    # Relations are links between versions, never merged by a shared author or topic.
    pmid_set = set(nodes)
    for node in nodes.values():
        node["versions"] = [r for r in node["related"] if r["pmid"] in pmid_set and r["type"] in {"UpdateIn", "UpdateOf", "RepublishedIn", "RepublishedFrom"}]
    write_json(output / "index.json", dict(schema_version=1, runs=runs))
    write_json(output / "network.json", dict(schema_version=1, nodes=list(nodes.values()), relationship_basis="Shared ontology concepts; author overlap is optional. Not a citation graph."))


def generate(papers, config, start, end, run_id, output, metadata=None):
    records = rank_records(papers, config)
    digest = dict(schema_version=1, topic="Chemosciences", run_id=run_id, window=dict(start=start, end=end),
                  generated_at=dt.datetime.now(dt.timezone.utc).isoformat(), ontology_version=config["version"],
                  ontology_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  scoring_note="Deterministic ontology signal, not scientific quality or predicted impact. No AI-generated summaries.",
                  search_metadata=metadata or {}, records=records)
    write_json(output / run_id / "digest.json", digest)
    selected = sorted([r for r in records if r["reading_list"]], key=lambda r: (not r["editor_pick"], r["editor_order"] or r["rank"]))
    md = [f"# Chemosciences: {start}–{end}", "", digest["scoring_note"], ""]
    for r in selected:
        p = r["paper"]
        md.extend([f"- [{p['title']}]({p['url']}) — {r['analysis']['score']}/100" + (" · PREPRINT" if p["preprint"] else ""), "  " + r["analysis"]["why"]])
    (output / run_id / "digest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    rebuild_catalog(output)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "chemosciences.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--run-id")
    parser.add_argument("--input-json", type=Path, help="Offline normalized article fixture for reproducible backfills.")
    parser.add_argument("--replace", action="store_true", help="Explicitly regenerate an existing historical run.")
    args = parser.parse_args()
    today = dt.datetime.now(dt.timezone.utc).date()
    last_sunday = today - dt.timedelta(days=today.weekday()+1)
    end = dt.date.fromisoformat(args.end) if args.end else last_sunday
    start = dt.date.fromisoformat(args.start) if args.start else end - dt.timedelta(days=6)
    if start > end or end >= today:
        parser.error("Use a completed date window with start <= end < today (UTC).")
    run_id = args.run_id or (end + dt.timedelta(days=1)).isoformat()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_id):
        parser.error("run-id must be YYYY-MM-DD")
    destination = args.output / run_id / "digest.json"
    if destination.exists() and not args.replace:
        print(f"Preserving existing run {run_id}; use --replace to regenerate explicitly.")
        rebuild_catalog(args.output)
        return
    config = read_json(args.config)
    if args.input_json:
        papers, metadata = read_json(args.input_json), dict(source="PubMed retrospective backfill", date_field="EDAT", offline=True)
    else:
        papers, metadata = retrieve(config, start.isoformat(), end.isoformat())
    digest = generate(papers, config, start.isoformat(), end.isoformat(), run_id, args.output, metadata)
    print(f"{run_id}: {len(digest['records'])} candidates; {sum(r['reading_list'] for r in digest['records'])} reading selections.")


if __name__ == "__main__":
    main()
