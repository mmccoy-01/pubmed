# Chemosciences Observatory

An independent weekly, static, ontology-ranked reading feed. No AI service,
API key, paid dependency, or notification destination is used. It searches
PubMed, including preprints already indexed there; it is not a complete search
of bioRxiv, medRxiv, arXiv, or Research Square.

The existing PhD query, scoring rules, workflow, output folders, and seen cache
are unchanged. The new job has the same `0 3 * * 1` UTC schedule and publishes
to **chemosciences-data**, so neither job races the other for the main branch.

## Run

```sh
python chemosciences_digest.py
python chemosciences_digest.py --start 2026-10-19 --end 2026-10-25
python -m unittest discover -s tests -p 'test_chemosciences*.py'
```

Python 3.12+ and its standard library are sufficient. The default window is
the last completed Monday–Sunday week in UTC, using PubMed Entry Date (EDAT).
The folder name is the following Monday. Re-running preserves an existing
weekly digest unless `--replace` is supplied explicitly. A custom `--output`
keeps experimental runs separate. Network/invalid-query failures happen before
any new digest is written. HTTP calls retry with backoff and respect the
unauthenticated NCBI request rate. Retrieval is paginated; oversized queries
and phrase warnings fail visibly instead of silently dropping candidates.

## Search and score

`config/chemosciences.json` contains eight overlapping query modules, versioned
ontology concepts and aliases, category caps, interaction bonuses, and selection
settings. Title matches receive twice the weight of abstract-only matches.
Aliases use word boundaries and optional `*` suffixes; repeating a word does
not increase its score. Domain points use the strongest domain, rather than
rewarding a paper for mentioning many different fields.

A candidate must match a sensing/adjacent domain and a biological question.
If the domain appears only in the abstract, multiple domain aliases and a
biological title signal are required. Corrections, comments, retracted records,
and nonbiological sensors without a matched biological mechanism are excluded
from selection. Weakly focused clinical and engineering papers can still slip
through; the rules are transparent and editable, not semantic understanding.

Scores have caps: domain 24, biology 36, mechanism 20, methods 8, interactions
12, maximum 100. No author, institution, journal, organism, or study-design
prestige multiplier exists. Scores measure **ontology signal**, not quality,
novelty, causal strength, or future influence. The config hash and version are
stored in each digest for reproducibility.

Up to five digest picks are chosen using score minus a small redundancy cost
for already represented biological concepts. At most one PubMed-labeled review
is included in the digest; other reviews remain eligible for the reading list.
Up to ten qualifying papers form that list. The rest remain in the candidate
archive. No filler is added when fewer qualify. The four initial backfills
use real PubMed records from the pilot and this deterministic scorer, **not**
the pilot's manually assigned editorial ratings or summaries.

## Outputs and website

`output/chemosciences/YYYY-MM-DD/digest.json` includes every fetched article,
score, matched terms, bonuses, original abstract, publication status, and
selection flags. `digest.md` is a readable export. `index.json` lists all
windows. `network.json` accumulates eligible papers with weeks, authors, and
concepts; it contains no AI-generated summaries. The website fetches the data
branch directly. A clearly labeled local four-week snapshot is available if
live data cannot be loaded.

The graph connects papers through shared, linkable biological concepts; a
single generic domain is insufficient. A specific mechanism or at least two
shared concepts creates a candidate edge. Inverse concept frequency orders
edges; each node keeps at most six strongest displayed edges. An optional
author-name layer is separate from biology. It is **not a citation graph**,
and author names are not identity-disambiguated. Nodes have equal size; digest
picks have an outline. At most 200 nodes are drawn at once, with an explicit
message asking users to narrow filters above that limit.

PMID/DOI duplicates are handled within a weekly run. Explicit PubMed version
relations are retained as metadata; preprint/publication pairs without reliable
metadata are not guessed or merged. Cross-week reappearances remain visible
as historical occurrences, not new independent studies. Full citation graphs,
fuzzy version reconciliation, automated scientific-quality assessment, and
complete external preprint coverage are outside this first version.

## Publication

The workflow must be committed to the repository's default branch for GitHub
to schedule it. Its first run creates `chemosciences-data`; later runs retain
that branch's archive and restore the current script, config, and tests from
the triggering branch before generating. It needs only the standard repository
`GITHUB_TOKEN` with contents-write permission supplied by Actions, not a user
token or a configured secret. It sends no notifications.
