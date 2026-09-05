# JATS-Parser-Baseline vor dem Refactoring

## Zweck

Diese Datei dokumentiert das Verhalten, die öffentliche Oberfläche und die
persistierten Artefaktschemata vor dem brechenden JATS-Parser-Refactoring. Die
zugehörigen maschinenlesbaren Golden-Snapshots liegen unter
`tests/fixtures/expected/` und werden von
`tests/test_jatsparser_baseline.py` geprüft.

Die Baseline beschreibt den Ausgangspunkt. Die bereits beschlossenen
Änderungen stehen in `JATSPARSER_REFACTORING_PLAN.md` und dürfen diese Werte in
späteren PRs ausdrücklich verändern.

## Ausgangspunkt

- Branch: `refactor/jats-parser-baseline`
- Basis-Commit: `5d445000f77664d7683e20f3025a93481df0a46d`
- Fixture-Anzahl: fünf JATS-Dokumente
- Testframework: Python `unittest`
- Referenzlauf am 4. September 2026: 98 Tests erfolgreich, ein Test
  übersprungen

## Aktuelle öffentliche Parseroberfläche

`JATSParser` wird über `pmcortex.JATSParser` exportiert. Die Methoden sind
technisch öffentlich, obwohl ein Teil davon einen zuvor aufgebauten internen
Zustand erwartet.

| Methode | Aktueller Vertrag und Zustandseffekt |
|---|---|
| `JATSParser()` | Erstellt den wiederverwendeten XML-Parser und setzt `last_context=None`. |
| `parse_article(path, pmcid=None, save_sentences=False)` | Hauptaufruf; setzt `root`, `namespaces`, `pmcid`, `contexts`, `sentences`, Diagnosen und `article`; gibt `JATSArticle` zurück. |
| `parse_tree(path)` | Liest XML und setzt `root` sowie `namespaces`; ist als `_Element` annotiert, gibt aktuell aber `None` zurück. |
| `extract_metadata()` | Erwartet `root` und `namespaces`; gibt `(title, abstract, pmcid, pmid)` zurück. |
| `extract_authors()` | Erwartet einen geparsten Baum; gibt Artikelautoren zurück. |
| `extract_references()` | Erwartet einen geparsten Baum; gibt `list[Reference]` zurück. |
| `extract_sections()` | Erwartet einen geparsten Baum; gibt lose Abschnittsdictionaries mit XML-Elementen zurück. |
| `extract_contexts(sections, references, save_sentences=False)` | Erwartet `pmcid`; mutiert Context-, Satz- und Diagnosezustand; gibt `None` zurück. |
| `metadata_to_dataframe()` | Erwartet `article`; erzeugt einen einzeiligen DataFrame. |
| `sources_to_dataframe()` | Erwartet `article`; erzeugt einen DataFrame aus Referenzen. |
| `contexts_to_dataframe()` | Erwartet `contexts`; erzeugt einen DataFrame aus Contexts. |
| `text_to_list()` | Erwartet `sentences`; rendert positionierte Sätze als Text. |
| `normalized_authors_and_refs()` | Erwartet `article`; erzeugt das Dictionary für das Autoren-YAML. |

### Repositoryinterne Aufrufer

- `README.md` verwendet `parse_article()` und die drei DataFrame-Methoden.
- `pmcortex.pipeline._parse_jats()` liest `article` sowie die Parserattribute
  `contexts`, `sentences`, `citation_diagnostics` und
  `unsafe_citation_rejections`.
- `tests/test_jatsparser.py` verwendet öffentliche und private Parsermethoden.
- `tests/test_citation_normalizer.py` verwendet den Parser als
  Integrationshülle für Normalisierer- und Satztests.
- Das aktuelle Notebook enthält keinen zusätzlichen direkten `JATSParser`-
  Aufruf, der über diese Inventarliste hinausgeht.

## Aktuelles XML- und Segmentierungsverhalten

Die Baseline hält folgende bekannten Grenzen ausdrücklich fest:

- `parse_tree()` benötigt einen Default-Namespace und wirft bei einem nur
  explizit präfixierten oder namespace-freien Root derzeit `KeyError`.
- `_split_sentences()` trennt an geeigneten Punkten, aber nicht an `?` oder
  `!`.
- Bei unterschiedlicher Satzanzahl von `text_with_markers` und
  `query_text_with_markers` wird der Context behalten und `query_raw=None`
  gesetzt.
- Zustandsabhängige Methoden vor `parse_article()` enden überwiegend mit
  `AttributeError`; nur `extract_contexts()` prüft die fehlende PMCID explizit
  und wirft `RuntimeError`.
- Eine wiederverwendete Parserinstanz ersetzt ihre artikelbezogenen Ergebnisse
  beim zweiten vollständigen Parse.

Diese Punkte sind keine Zielanforderungen. Sie werden nur festgehalten, damit
die späteren, bewusst brechenden Änderungen sichtbar bleiben.

## Golden-Snapshot-Schema

Jede Datei `tests/fixtures/expected/PMC....json` enthält:

| Schlüssel | Inhalt |
|---|---|
| `metadata` | PMCID, PMID, Titel, Abstract und Artikelautoren |
| `references` | alle extrahierten `Reference`-Felder in Dokumentreihenfolge |
| `sections` | Titel, hierarchischer Titel und Paragraphenzahl in aktueller Postorder |
| `contexts` | sämtliche `Context`-Felder einschließlich Strukturposition, Query, Hits und Vorgängercontext |
| `sentences` | gespeicherte `PositionedSentence`-Werte einschließlich Markertext |
| `normalized_authors_and_refs` | aktuelle Autoren- und Zitationsautorenlisten |
| `citation_diagnostics` | Diagnosecode, RIDs und XML-Quellzeile |
| `unsafe_citation_rejection_count` | Anzahl verworfener semantisch unsicherer Queries |

Die aktuellen fünf kompakten Fixtures erzeugen jeweils einen Abschnitt, eine
Referenz, einen Context und einen Satz. Sie decken deshalb die vollständige
Form der Datenmodelle ab, aber nicht jede komplexe Zitierungsvariante; dafür
bleiben die spezialisierten Normalisierertests maßgeblich.

## Aktuelle CSV-Artefakte

Die Typangaben unterscheiden den logischen Python-Wert vom beobachteten
Pandas-Dtype. Optionale Felder können bei anderen Artikeln oder leeren Tabellen
zu einem anderen von Pandas inferierten Dtype führen. Maßgeblich sind daher
Spaltenreihenfolge und logischer Typ.

### `metadata.csv`

| Position | Spalte | Logischer Typ | Beobachteter Dtype |
|---:|---|---|---|
| 1 | `pmcid` | `str` | `str` |
| 2 | `pmid` | `str | None` | `str` |
| 3 | `title` | `str | None` | `str` |
| 4 | `abstract` | `str | None` | `str` |
| 5 | `authors` | `list[str]` | `object` |

### `sources.csv`

| Position | Spalte | Logischer Typ | Beobachteter Dtype |
|---:|---|---|---|
| 1 | `rid` | `str | None` | `str` |
| 2 | `label` | `str | None` | `str` |
| 3 | `doi` | `str | None` | `str` |
| 4 | `pmid` | `str | None` | `str` |
| 5 | `pmcid` | `str | None` | `object` |
| 6 | `year` | `int | None` | `int64` |
| 7 | `title` | `str | None` | `str` |
| 8 | `journal` | `str | None` | `str` |
| 9 | `authors` | `list[str]` | `object` |
| 10 | `source_pmcid` | `str` | `str` |

### `contexts.csv`

| Position | Spalte | Logischer Typ | Beobachteter Dtype |
|---:|---|---|---|
| 1 | `source_pmcid` | `str` | `str` |
| 2 | `section_index` | `int` | `int64` |
| 3 | `paragraph_index` | `int` | `int64` |
| 4 | `sentence_index` | `int` | `int64` |
| 5 | `query_raw` | `str | None` | `str` |
| 6 | `query` | `str` | `str` |
| 7 | `citation_forms` | `list[str]` | `object` |
| 8 | `citation_cleanup_action` | `str` | `str` |
| 9 | `hits` | `list[str]` | `object` |
| 10 | `query_length` | `int` | `int64` |
| 11 | `n_hits` | `int` | `int64` |
| 12 | `context` | `str | None` | `object` |

Listen werden von Pandas derzeit als textuelle Python-Repräsentation in CSV
geschrieben. Die spätere Serialisierungsphase muss dieses bestehende Verhalten
entweder bewusst erhalten oder versioniert ersetzen.

## Weitere persistierte Parserartefakte

### Fulltext

Aktuelles Zeilenformat:

```text
<section_index>/<paragraph_index>/<sentence_index><TAB><query_text_with_markers>
```

Die PMCID steht nur im Dateinamen. Der Text ist entgegen der Bezeichnung nicht
der sichtbare Originalsatz, sondern die markertragende Query-Darstellung.

### Autoren-YAML

```yaml
authors:
  - Smith J
cited_authors:
  - - Doe J
```

### Status-JSON

Der Parser beeinflusst insbesondere folgende Felder des Pipeline-Status:

- `parser_schema_version`,
- `diagnostic_count`,
- `unsafe_citation_rejection_count`.

## Geplante bewusste Baseline-Abweichungen

Spätere PRs dürfen nach den bestätigten Entscheidungen insbesondere ändern:

- sofortiger Ersatz der zustandsbehafteten Parser-API,
- Tupel und typisierte Abschnitts-/Ergebnisdataclasses,
- Preorder statt Postorder für verschachtelte Abschnitte,
- Unterstützung aller drei Namespace-Varianten,
- Satzgrenzen an `?` und `!`,
- Verwerfen statt Behalten nicht ausrichtbarer Contexts,
- sichtbarer Originalsatz statt Query-Darstellung im Fulltext,
- gemeinsame globale `sentence_id` in Fulltext und Context-Tabelle,
- zusätzliche Parser- und Recovery-Diagnosen.

Jede solche Abweichung muss mit einer neuen Erwartung, einer Begründung im
zugehörigen PR und gegebenenfalls einer neuen `parser_schema_version`
einhergehen.
