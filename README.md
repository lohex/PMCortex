# PMCortex

PMCortex is a research toolkit for building **biomedical document retrieval
datasets** from articles in [PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/).

The central idea is to turn a sentence containing one or more citations into a
retrieval query. The papers cited in that sentence are the relevant documents.
Other papers cited elsewhere in the same source article—but not in the query
sentence—provide particularly useful **hard negatives**.

These negatives are harder than randomly sampled papers because they usually
share the source article's topic, terminology, and scientific context with the
positive documents. A retrieval model must therefore distinguish papers that
are broadly related from those that specifically support the query sentence.

PMC download and JATS parsing are supporting parts of this workflow, not the
main purpose of the project.

## Retrieval task

PMCortex derives a retrieval example from a source article as follows:

```text
Source article
  │
  ├── Sentence with citations ──► query
  │                                │
  │                                └── papers cited in this sentence
  │                                    = relevant documents
  │
  └── Papers cited elsewhere in the same article
      = hard-negative candidates
```

For example, if a sentence cites references `[3, 7]`, its text becomes the
query and the papers behind references 3 and 7 become its positive documents.
References cited by the same article but absent from that sentence can be used
as hard negatives.

This construction has two useful properties:

- **Citation-grounded relevance:** positive documents are explicitly linked to
  the query statement by the article's authors.
- **Topically close negatives:** negative candidates come from the same
  bibliography and are therefore much less trivial than random corpus
  negatives.

PMCortex also supports selecting a shared subset of references for multiple
queries. This makes it possible to create controlled candidate pools containing
both positives and reusable hard negatives.

## Features

- extract retrieval queries, positive documents, and hard-negative candidates
  from citation contexts
- filter queries by citation count, length, self-citation, explicit answer
  leakage, and context-dependent wording
- construct candidate pools and shared-reference decoy sets
- download PMC articles as JATS/NXML through the OAI-PMH API
- parse article metadata, bibliographies, sections, and citation contexts
- fetch PubMed abstracts and MeSH assignments through NCBI E-utilities
- inspect local MeSH and Gene Ontology hierarchies

## Repository structure

```text
.
├── src/pmcortex/
│   ├── jatsparser.py         # metadata, references, and citation contexts
│   ├── query_filters.py      # query filtering and hard-negative selection
│   ├── downloader.py         # PMC JATS downloads through OAI-PMH
│   ├── mesh.py               # PubMed abstracts and MeSH annotations
│   ├── go_terms.py           # Gene Ontology OBO processing
│   ├── pmcsearch.py          # basic PMC website search
│   └── models.py             # pipeline data classes
├── data/
│   ├── desc2026.xml          # MeSH descriptor data
│   └── go.obo.obo            # Gene Ontology data
├── notebooks/
│   ├── PMCortex.ipynb        # exploratory dataset construction
│   └── EvaluateOnQueries.ipynb
├── tests/
├── pyproject.toml
└── uv.lock
```

## Installation

Requirements:

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) is recommended
- internet access for PMC and PubMed requests

Clone the repository and install the locked dependencies:

```bash
git clone <repository-url>
cd PMCortex
uv sync
```

Alternatively, install the package in editable mode with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Dataset

`data/pmc_context_queries_v0-5/` contains a generated snapshot of the retrieval
dataset. The three central intermediate tables are:

- `extracted_contexts.csv`: query sentences, citation hits, and their source
  context
- `extracted_metadata.csv`: metadata for the PMC source articles
- `extracted_sources.csv`: bibliography entries extracted from those articles

The benchmark CSV files contain later processing stages, including filtered
queries and alternative decoy constructions. They allow work on filtering and
evaluation without downloading and parsing all source articles again.

## Quick start

### Parse an article into retrieval data

```python
from pmcortex import JATSParser

parser = JATSParser()
article = parser.parse_article("data/pmc_jats/PMC5513360.nxml")

metadata = parser.metadata_to_dataframe()
sources = parser.sources_to_dataframe()
contexts = parser.contexts_to_dataframe()

print(contexts[["query", "hits"]].head())
```

The resulting tables represent:

- `metadata`: the source article's PMCID, PMID, title, abstract, and authors
- `sources`: documents in the source article's bibliography, including their
  DOI, PMID, and authors
- `contexts`: sentence-derived queries and the bibliography references cited
  by each query

For a given row in `contexts`, `hits` identifies the positive references.
Other entries in `sources` from the same `source_pmcid` are potential hard
negatives, provided they are not cited in that query sentence.

### Filter retrieval queries

`QueryFilterPipeline` expects `extracted_contexts.csv`,
`extracted_metadata.csv`, and `extracted_sources.csv` in the supplied
directory.

```python
from pmcortex import QueryFilterPipeline

pipeline = QueryFilterPipeline("data/pmc_context_queries_v0-5")

(
    pipeline
    .filter_min_hit_queries(1)
    .filter_no_et_al_queries()
    .filter_no_explicit_spoilers()
    .filter_no_self_citations()
    .filter_min_source_count(2)
    .filter_min_query_length(8)
    .filter_max_query_length(60)
    .filter_no_anaphoric_starts()
    .filter_no_reference_framing_starts()
    .filter_no_summary_starts()
)

queries = pipeline.filtered_df
print(queries[["query", "hits"]].head())
```

The methods update `filtered_df` step by step and return the pipeline itself,
so they can be chained. Expensive annotations are computed lazily when a
filter first needs them. `reset()` restores the unfiltered contexts.

`select_shared_reference_subset(...)` can then select a controlled set of
same-article references for use as decoys across multiple queries. Positive
references are excluded from the corresponding query's negative options.

### Download PMC JATS files

```python
from pmcortex.downloader import PMCJATSDownloader

with PMCJATSDownloader(
    "data/pmc_jats",
    user_agent="MyProject/1.0 (contact: name@example.org)",
) as downloader:
    results = downloader.download_many(["PMC5513360", "PMC11587633"])

for result in results:
    print(result.pmcid, result.status, result.path)
```

Existing files are reused by default. The downloader applies request-rate
limiting and retries temporary failures with exponential backoff.

### Fetch abstracts and MeSH annotations

```python
from pmcortex import PubMedMeSHClient

with PubMedMeSHClient(
    user_agent="MyProject/1.0 (contact: name@example.org)"
) as client:
    abstracts = client.fetch_abstracts(["12345678", "23456789"])
    mesh_terms = client.fetch_mesh_terms(["12345678", "23456789"])
```

The client limits requests to fewer than three per second by default. Larger
PMID collections should be divided into appropriate batches.

## End-to-end data flow

```text
PMC IDs
  │
  ▼
OAI-PMH download
  │
  ▼
JATS articles
  │
  ▼
Metadata + bibliography + citation contexts
  │
  ├── cited in query sentence ─────────► positive documents
  │
  └── cited elsewhere in source paper ─► hard-negative candidates
  │
  ▼
Query filtering and candidate-pool construction
  │
  ▼
Document retrieval benchmark
```

## Tests

Run the test suite with:

```bash
uv run python -m unittest discover -s tests
```

Run a single test module with:

```bash
uv run python -m unittest tests.test_query_filters
```

Most network access is replaced by test doubles. The parser integration tests
additionally expect five JATS example files in `data/pmc_jats/`; the expected
PMCIDs are listed in `tests/_shared.py`. These files are not included in the
current repository snapshot, so that part of the test suite fails when they
are absent.

## Operational notes

- Set a descriptive `User-Agent` containing a contact address when accessing
  NCBI services.
- `JATSParser` is a local parser and does not perform network requests.
- JATS documents vary in structure, so the parser uses fault-tolerant XML
  processing.
- Runtime information is written through `loguru` to
  `logs/pmcortex.log`.
- `pmcsearch.py` parses the PMC website's HTML and is therefore more sensitive
  to website changes than the API-based components.

## Project status

PMCortex is an early-stage research project (`0.1.0`). Its public API and data
schemas may still change. The repository currently does not specify a license.
