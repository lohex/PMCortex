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
- discover PMC seed articles through NCBI ESearch and ESummary
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
│   ├── pmcsearch.py          # PMC search through ESearch and ESummary
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

New builds use recoverable per-article storage before producing these combined
tables:

```text
pmc_context_queries_v1/
├── xml_jats/                 # temporary or retained parser inputs
├── parsed/PMC.../            # metadata, sources, contexts, diagnostics
├── fulltexts/PMC....txt      # positioned sentences
├── authors/PMC....yaml
├── status/PMC....json        # success or failure reason per article
├── extracted_metadata.csv    # materialized combined tables
├── extracted_sources.csv
└── extracted_contexts.csv
```

## Quick start

### Parse an article into retrieval data

```python
from pmcortex import (
    JATSParser,
    contexts_to_dataframe,
    metadata_to_dataframe,
    references_to_dataframe,
)

result = JATSParser().parse(
    "data/pmc_jats/PMC5513360.nxml",
    expected_pmcid="PMC5513360",
    include_sentences=True,
)

metadata = metadata_to_dataframe(result.article)
sources = references_to_dataframe(
    result.article.pmcid,
    result.article.references,
)
contexts = contexts_to_dataframe(result.contexts)

print(contexts[["query", "hits"]].head())
```

The resulting tables represent:

- `metadata`: the source article's PMCID, PMID, title, abstract, and authors
- `sources`: documents in the source article's bibliography, including their
  DOI, PMID, and authors
- `contexts`: sentence-derived queries and the bibliography references cited
  by each query, together with `source_pmcid`, `section_index`,
  `paragraph_index`, and `sentence_index`

Generated full-text files contain one sentence per line. Every line starts
with the same structural indices used by the query table:

```text
0/2/1	Sentence text...
```

Every article shard also contains `diagnostics.json`. It preserves detailed
citation-normalization issues and XML repairs reported by libxml2; the status
record contains their total count.

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
    .filter_no_citation_spoilers()
    .filter_no_self_citations()
    .filter_min_source_count(2)
    .filter_min_query_length(8)
    .filter_max_query_length(60)
    .filter_no_anaphoric_starts()
    .filter_no_reference_framing_starts()
    .filter_no_summary_starts()
)

pipeline.assert_no_citation_spoilers()

queries = pipeline.filtered_df
print(queries[["query", "hits"]].head())
```

The methods update `filtered_df` step by step and return the pipeline itself,
so they can be chained. Expensive annotations are computed lazily when a
filter first needs them. `reset()` restores the unfiltered contexts.

`select_shared_reference_subset(...)` can then select a controlled set of
same-article references for use as decoys across multiple queries. Positive
references are excluded from the corresponding query's negative options.

### Filter cross-context coreferences

The deterministic self-contained filter removes explicit document references,
but deliberately keeps ambiguous pronouns. An optional second stage can remove
only queries whose anaphoric mention resolves to the preceding context:

```python
import spacy
from fastcoref import spacy_component

from pmcortex import CrossContextCoreferenceFilter

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")

coreference_filter = CrossContextCoreferenceFilter(nlp)
filtered_queries = (
    coreference_filter
    .filter_cross_context_coreferences(queries)
)
```

Model installation and configuration are explicit; the filter does not
download language models or mutate the input DataFrame.

### Discover PMC seed articles

`PMCSearch.search()` returns its results directly as a DataFrame. `max_results`
requests up to that many relevance-ordered records; fewer rows are returned
when PMC has fewer matches.

```python
from pmcortex import PMCSearch

with PMCSearch(
    user_agent="MyProject/1.0 (contact: name@example.org)",
) as search:
    seeds = search.search("breast cancer", max_results=100)

print(seeds[["pmcid", "title"]])
```

Search identifiers come from ESearch, and titles are fetched through ESummary
in rate-limited batches. PMC ESearch supports at most 10,000 results for one
query.

### Download and parse PMC articles

```python
from pathlib import Path

from pmcortex import PMCIngestionPipeline
from pmcortex.downloader import PMCJATSDownloader

dataset_root = Path("data/pmc_context_queries_v1")
with PMCJATSDownloader(
    dataset_root / "xml_jats",
    user_agent="MyProject/1.0 (contact: name@example.org)",
) as downloader:
    ingestion = PMCIngestionPipeline(
        dataset_root,
        downloader,
        parser_workers=4,
        delete_jats_after_success=True,
        parser_schema_version="citation-normalization-v3-no-spoilers",
        verbose=False,
    )
    records = ingestion.run(["PMC5513360", "PMC11587633"])
    tables = ingestion.materialize_tables()

for record in records:
    print(record.pmcid, record.status, record.reason)
```

The downloader remains serial so its NCBI rate limit applies globally within
the run, while already downloaded articles are parsed in a bounded process
pool. Per-article files are written atomically by the main process and the
combined tables are rebuilt from complete shards.

With `delete_jats_after_success=True`, a JATS file is removed only after its
metadata, sources, contexts, positioned full text, author data, and successful
status record have all been stored. Parser and persistence failures keep the
JATS input for diagnosis and retry. Cleanup is disabled by default.

Every outcome is independently auditable under `status/`. The JSON `status`
is one of `complete`, `not_found`, `no_fulltext`, `download_error`,
`parse_error`, or `persist_error`; `reason` contains the concrete error. The
same information is available programmatically:

```python
failed = [
    record
    for record in ingestion.list_status_records()
    if record.status != "complete"
]
```

Calling `run()` again reuses complete artifacts even when their JATS sources
were deleted. Changing `parser_schema_version` invalidates old outputs; the
missing source is then downloaded and parsed again. `parse_available()` parses
only local JATS files and performs no network requests.

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

Most network access is replaced by test doubles. Small synthetic JATS documents
under `tests/fixtures/jats/` exercise the parser integration path without
requiring downloaded PMC articles.

## Operational notes

- Set a descriptive `User-Agent` containing a contact address when accessing
  NCBI services.
- `JATSParser` is a local parser and does not perform network requests.
- JATS documents vary in structure, so the parser uses fault-tolerant XML
  processing.
- Ingestion shows one progress bar. Set `verbose=True` for detailed pipeline
  messages; runtime logs are also written to `logs/pmcortex.log`.
- `PMCSearch` uses the PMC ESearch and ESummary APIs with rate limiting and
  retries; PMC may still return fewer rows than requested for a query.

## Project status

PMCortex is an early-stage research project (`0.1.0`). Its public API and data
schemas may still change. The repository currently does not specify a license.
