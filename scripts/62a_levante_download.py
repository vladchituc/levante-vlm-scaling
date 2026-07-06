#!/usr/bin/env python3
"""
62a_levante_download.py — Mirror the public LEVANTE-bench results bucket locally.

Source: gs://levante-bench (public, no auth; JSON API via storage.googleapis.com).
Paper: LEVANTE-bench, arXiv 2606.05497 (Tan ... Frank). Repo:
anonymous.4open.science/r/levante-bench-3013 (needs browser user-agent).

Downloads, into data/levante-bench/ :
  comparison/   all results/comparison/*_accuracy.csv  (schema: task, model,
                item_uid, correct, difficulty) — difficulty = child-calibrated
                Rasch item easiness/difficulty parameter exported by the authors.
  v1/<model>/<run>/  per-task trial CSVs (+ metadata.json, summary.csv) for
                English base models only (no -de/-es suffix). Skips .npy,
                responses.json, -by-type.csv (not needed for theta fits).

Idempotent: skips files that already exist with nonzero size.
Writes data/levante-bench/62a_download_manifest.json (counts + failures).

Usage: python3 scripts/62a_levante_download.py
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "levante-bench"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
DEST = os.path.join(os.path.dirname(__file__), "..", "data", "levante-bench")
WORKERS = 12

KEEP_BASENAMES = {"metadata.json", "summary.csv"}
# Item banks (stimulus text, trial_type, chance_level, child IRT params where
# published) — needed for subtype breakdowns (62e).
CORPUS_FILES = [
    "corpus_data/v1/corpus/egma-math/math-item-bank.csv",
    "corpus_data/v1/corpus/trog/trog-item-bank-full-params.csv",
    "corpus_data/v1/corpus/theory-of-mind/theory-of-mind-item-bank.csv",
    "corpus_data/v1/corpus/mental-rotation/mental-rotation-item-bank.csv",
    "corpus_data/v1/translations/item-bank-translations.csv",
]
TASK_CSVS = {
    "egma-math.csv", "matrix-reasoning.csv", "mental-rotation.csv",
    "theory-of-mind.csv", "trog.csv", "vocab.csv",
}


def list_objects(prefix):
    """List all object names under a prefix (handles pagination)."""
    names, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(API + "?" + urllib.parse.urlencode(q)) as r:
            d = json.load(r)
        names += [it["name"] for it in d.get("items", []) if int(it["size"]) > 0]
        token = d.get("nextPageToken")
        if not token:
            return names


def want(name):
    """Filter: comparison accuracy CSVs + English-model run files."""
    if name.startswith("results/comparison/"):
        return name.endswith("_accuracy.csv")
    if name.startswith("results/v1/"):
        parts = name.split("/")
        if len(parts) < 4:
            return False
        model = parts[2]
        if model.endswith("-de") or model.endswith("-es"):
            return False
        base = parts[-1]
        return base in KEEP_BASENAMES or base in TASK_CSVS
    return False


def fetch(name):
    rel = (name.replace("results/", "", 1) if name.startswith("results/")
           else name.replace("corpus_data/v1/", "corpus/", 1))
    dest = os.path.join(DEST, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return ("skipped", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = (f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/"
           + urllib.parse.quote(name, safe="") + "?alt=media")
    try:
        with urllib.request.urlopen(url) as r, open(dest + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dest + ".part", dest)
        return ("ok", name)
    except Exception as e:  # noqa: BLE001 — record and continue
        return ("fail", f"{name}: {e}")


def main():
    os.makedirs(DEST, exist_ok=True)
    names = [n for n in list_objects("results/comparison/") + list_objects("results/v1/")
             if want(n)] + CORPUS_FILES
    print(f"{len(names)} files to mirror into {os.path.abspath(DEST)}")
    tallies = {"ok": 0, "skipped": 0, "fail": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed(ex.submit(fetch, n) for n in names):
            status, detail = fut.result()
            tallies[status] += 1
            if status == "fail":
                failures.append(detail)
    manifest = {"n_listed": len(names), **tallies, "failures": failures}
    with open(os.path.join(DEST, "62a_download_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(manifest if not failures else json.dumps(manifest, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
