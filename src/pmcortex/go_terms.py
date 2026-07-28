from __future__ import annotations

from pathlib import Path
from typing import Literal

from goatools.obo_parser import GODag


def get_go_terms_up_to_level(
    obo_path: str | Path,
    max_level: int,
    *,
    namespace: Literal["biological_process", "molecular_function", "cellular_component"] | None = None,
    include_obsolete: bool = False,
    use_depth: bool = False,
) -> list[dict[str, str | int | None]]:
    """
    Return GO terms up to a given level/depth, including definitions.

    Parameters
    ----------
    obo_path
        Path to go.obo
    max_level
        Maximum GO level (or depth, if use_depth=True) to include
    namespace
        Optional namespace filter:
        - biological_process
        - molecular_function
        - cellular_component
    include_obsolete
        Whether to include obsolete GO terms
    use_depth
        If False, filter by `term.level` (minimum distance to root).
        If True, filter by `term.depth` (maximum distance to root).

    Returns
    -------
    list[dict]
        Sorted list of dicts with keys:
        - go_id
        - name
        - definition
        - namespace
        - level
        - depth
        - is_obsolete
    """
    if max_level < 0:
        raise ValueError("max_level must be >= 0")

    go_dag = GODag(str(obo_path), optional_attrs={"def", "synonym"})
    results: list[dict[str, str | int | None]] = []

    for go_id, term in go_dag.items():
        term_namespace = getattr(term, "namespace", None)
        term_definition = getattr(term, "defn", None)
        term_is_obsolete = bool(getattr(term, "is_obsolete", False))
        term_level = getattr(term, "level", None)
        term_depth = getattr(term, "depth", None)

        if namespace is not None and term_namespace != namespace:
            continue

        if not include_obsolete and term_is_obsolete:
            continue

        metric = term_depth if use_depth else term_level
        if metric is None or metric > max_level:
            continue

        results.append(
            {
                "go_id": go_id,
                "name": term.name,
                "definition": term_definition,
                "namespace": term_namespace,
                "level": term_level,
                "depth": term_depth,
                "is_obsolete": term_is_obsolete,
            }
        )

    results.sort(key=lambda x: (x["namespace"] or "", int(x["level"] or 0), str(x["go_id"])))
    return results