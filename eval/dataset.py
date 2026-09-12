import json
from pathlib import Path

DEFAULT_PATH = Path("eval/eval_set.json")


def load_eval_set(path: Path = DEFAULT_PATH) -> list[dict]:
    """Expects a JSON list of objects:
    {"question": str, "ground_truth": str, "retrieved_contexts": list[str]}
    Does not assume the file exists yet, raises with a clear message if missing."""
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



