# # Gustatory Development & Signaling Digest

Forked from https://github.com/felizvida/pubmed

A personalized, ontology-driven literature digest focused on gustatory system biology, developmental signaling, Hedgehog pathway regulation, tissue homeostasis, and epithelial sensory organ maintenance.

This project is a fork and extension of the original PubMed digest workflow, customized for:

* developmental biology
* taste organ research
* SHH/Wnt/BMP signaling
* stem cell niche maintenance
* epithelial–mesenchymal interactions
* mechanistic and perturbation-based studies

## Features

* Weekly automated GitHub Actions workflow
* PubMed + bioRxiv integration
* Local ontology-based scoring **(no OpenAI API key required)**
* JSON + Markdown digest generation
* Persistent seen-paper caching
* Modern web dashboard integration
* Editor digest highlighting
* Per-paper RIS/Zotero export
* Preprint detection and labeling

## Dashboard

The repository includes a modern dashboard for browsing weekly digests:

```text
Gustatory Development & Signaling Digest
```

Features include:

* weekly dropdown navigation
* ontology signal visualization
* expandable paper analysis
* editor picks
* clickable PubMed links
* AI-ready prompt copying
* source and preprint labeling

## Workflow

Each weekly run:

1. Queries literature sources
2. Scores papers against a custom ontology
3. Generates:

   * `digest.md`
   * `digest.json`
4. Publishes outputs to the website dashboard

## Structure

```text
topics/                 # reusable PubMed query files
config/                 # ontology scoring rules
output/YYYY-MM-DD/      # weekly generated digests
.github/workflows/      # automated weekly runs
```

## Notes

This repository prioritizes:

* mechanistic biology
* developmental logic
* morphogen signaling
* in vivo perturbation studies
* bridge papers connecting development and adult tissue homeostasis

rather than generic oncology or purely descriptive studies.
