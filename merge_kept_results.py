#!/usr/bin/env python3
import json
from pathlib import Path
import sys


def load_json(path: Path):
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # Some files might be an object with a top-level key
            for v in data.values() if isinstance(data, dict) else []:
                if isinstance(v, list):
                    return v
            return []
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return []


def main():
    base = Path(__file__).parent
    f1 = base / "FINAL" / "gpt5_mini_labeled_kept_results.json"
    f2 = base / "FINAL" / "gpt5_mini_labeled_new_kept_results.json"
    out = base / "data.json"

    a = load_json(f1)
    b = load_json(f2)

    print(f"Loaded {len(a)} records from {f1.name}")
    print(f"Loaded {len(b)} records from {f2.name}")

    merged = {}
    # Policy: start with the older file, then overlay/replace with newer file entries
    def key_for(rec):
        if not isinstance(rec, dict):
            return None
        k = rec.get("article_url") or rec.get("url") or rec.get("id")
        if k:
            return str(k).strip()
        # fallback to full-record hash (stable)
        try:
            return json.dumps(rec, sort_keys=True, ensure_ascii=False)
        except Exception:
            return None

    for rec in a:
        k = key_for(rec)
        if k is None:
            continue
        merged[k] = rec

    for rec in b:
        k = key_for(rec)
        if k is None:
            continue
        # newer file overrides older entries with same key
        merged[k] = rec

    merged_list = list(merged.values())

    total_input = len(a) + len(b)
    unique = len(merged_list)
    duplicates_removed = total_input - unique

    try:
        with out.open("w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=2)
        print(f"Wrote {unique} unique records to {out}")
        print(f"Total input records: {total_input}")
        print(f"Duplicates removed: {duplicates_removed}")
    except Exception as e:
        print(f"Failed to write output file: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
