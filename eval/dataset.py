import json
from pathlib import Path

from src.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "eval" / "eval_set.json"


def load_eval_set(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"no eval set at {path}. create a JSON list of "
            '{"question", "ground_truth", "retrieved_contexts"} entries first.'
        )
    with path.open() as f:
        records = json.load(f)

    required_keys = {"question", "ground_truth", "retrieved_contexts"}
    for i, record in enumerate(records):
        missing = required_keys - record.keys()
        if missing:
            raise ValueError(f"eval set record {i} missing keys: {missing}")

    return records
