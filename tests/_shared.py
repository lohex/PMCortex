from pathlib import Path

EXAMPLE_PMCIDS = [
    "PMC3438321",
    "PMC440378",
    "PMC2693326",
    "PMC10139131",
    "PMC10462751",
    "PMC11614679",
]

DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "jats"


def existing_example_files() -> dict[str, Path]:
    return {pmcid: DATA_DIR / f"{pmcid}.nxml" for pmcid in EXAMPLE_PMCIDS if (DATA_DIR / f"{pmcid}.nxml").exists()}
