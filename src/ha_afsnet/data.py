"""Released chronological splits, article identifiers and causal graph statistics."""

import argparse
import csv
import json
import os
import tempfile
import zipfile
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, PROJECT_ROOT
from .utils import dump, manifest_fingerprint, sha256


class RegionalData:
    def __init__(self, region, device="cuda:0", data_root=None, prepared_root=None):
        self.region, self.device = region, torch.device(device)
        base = Path(data_root or DEFAULT_DATA_ROOT)
        prepared = Path(prepared_root or DEFAULT_PREPARED_ROOT)
        prepared.mkdir(parents=True, exist_ok=True)
        self.data_root, self.prepared_root = base, prepared
        source = base / "data" / region
        self.n, self.r = map(int, (source / "stat.txt").read_text().split())
        self.heads = set(json.loads((source / "head_ents.json").read_text()))
        manifest = json.loads((prepared / region / "manifest.json").read_text())
        reference = json.loads((PROJECT_ROOT / "splits" / f"{region}.json").read_text())
        self.input_fingerprint = manifest_fingerprint(manifest)
        if self.input_fingerprint != manifest_fingerprint(reference):
            raise ValueError("Prepared split differs from the bundled data reference")
        self.compatible_manifest_hashes = {
            sha256(prepared / region / "manifest.json"),
            reference.get("reference_manifest_sha256"),
        }
        if (self.n, self.r) != (
            manifest["entities"],
            manifest["relations"],
        ) or self.heads != set(manifest["popularity"]["excluded_targets"]):
            raise ValueError("Vocabulary or popularity differs from split manifest")
        self.dates = json.loads((prepared / region / "dates.json").read_text())
        self.splits, self.rows, self.graph_cache = {}, {}, OrderedDict()
        self.answer_cache = OrderedDict()
        for split in ["train", "valid", "test"]:
            path = (
                base
                / "data_disentangled"
                / f"{region}_LDA_K5"
                / f"{split}_w_contextid.txt"
            )
            cache = prepared / region / f"{split}.npy"
            cache_meta = cache.with_suffix(".cache.json")
            fingerprint = sha256(path)
            if (
                manifest["source_sha256"].get(
                    f"data/SeCoGD/data_disentangled/{region}_LDA_K5/{split}_w_contextid.txt"
                )
                != fingerprint
            ):
                raise ValueError(
                    f"{path} differs from released split; investigate the data change before rerunning prepare"
                )
            metadata = json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
            if (
                not cache.exists()
                or metadata.get("source_sha256") != fingerprint
                or metadata.get("array_sha256") != sha256(cache)
            ):
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=cache.parent,
                        prefix=cache.stem + ".",
                        suffix=".npy",
                        delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        np.save(stream, np.loadtxt(path, dtype=np.int64, ndmin=2))
                    os.replace(temporary, cache)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
                dump(
                    cache_meta,
                    {"source_sha256": fingerprint, "array_sha256": sha256(cache)},
                )
            array = np.load(cache, mmap_mode="r")
            if array.shape != (manifest["splits"][split]["rows"], 5):
                raise ValueError("Cached split dimensions differ from released source")
            times, starts = np.unique(array[:, 3], return_index=True)
            self.splits[split] = times.tolist()
            for t, start, end in zip(times, starts, [*starts[1:], len(array)]):
                self.rows[int(t)] = np.asarray(array[start:end])
        self.activity, self.row_activity, self.composition, self.rho = self.statistics()

    def statistics(self):
        total = max(self.rows) + 1
        activity, row_activity = np.zeros(total), np.zeros(total)
        compositions, rho = np.zeros((total, 5)), np.zeros(total + 1)
        ema = None
        for t in range(total):
            rows = self.rows.get(t, np.empty((0, 5), dtype=np.int64))
            active = np.unique(rows[:, [0, 2]]).size
            edges = np.unique(rows[:, :3], axis=0)
            activity[t] = np.log1p(active + len(edges))
            row_activity[t] = np.log1p(active + len(rows))
            counts = np.array(
                [len(np.unique(rows[rows[:, 4] == k, :3], axis=0)) for k in range(5)]
            )
            p = (counts + 1e-8) / (counts.sum() + 5e-8)
            compositions[t] = p
            rho[t + 1] = 0.0 if ema is None else float(np.sum(p * np.log(p / ema)))
            ema = p if ema is None else 0.9 * ema + 0.1 * p
        return activity, row_activity, compositions, rho

    def history(self, t, width):
        assert t > 0
        return list(range(max(0, t - width), t))

    def answers(self, t, contextual=False):
        key = (t, contextual)
        if key in self.answer_cache:
            self.answer_cache.move_to_end(key)
            return self.answer_cache[key]
        answers = defaultdict(set)
        for s, r, o, _, k in self.rows[t]:
            answers[(int(s), int(r), int(k)) if contextual else (int(s), int(r))].add(
                int(o)
            )
        self.answer_cache[key] = answers
        while len(self.answer_cache) > 48:
            self.answer_cache.popitem(last=False)
        return answers


def prepare_region(region, data_root=None, prepared_root=None):
    base = Path(data_root or DEFAULT_DATA_ROOT)
    source = base / "data" / region
    contexts = base / "data_disentangled" / f"{region}_LDA_K5"
    out = Path(prepared_root or DEFAULT_PREPARED_ROOT) / region
    out.mkdir(parents=True, exist_ok=True)
    dates = {}
    with (source / f"{region}.csv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t, date = int(row["timid"]), row["date"]
            assert t not in dates or dates[t] == date
            dates[t] = date
    assert sorted(dates) == list(range(max(dates) + 1)), "Missing timestamp IDs"
    calendar = [datetime.strptime(dates[t], "%Y%m%d") for t in sorted(dates)]
    assert all((b - a).days == 1 for a, b in zip(calendar, calendar[1:])), (
        "Snapshots are not contiguous calendar days"
    )
    n, r = map(int, (source / "stat.txt").read_text().split())
    summary = {
        "dataset": region,
        "entities": n,
        "relations": r,
        "days": len(dates),
        "date_min": min(dates.values()),
        "date_max": max(dates.values()),
        "splits": {},
        "source_sha256": {},
    }
    frequencies = Counter()
    articles = {}
    split_ids = []
    for split in ["train", "valid", "test"]:
        path = source / f"{split}.txt"
        contextual = contexts / f"{split}_w_contextid.txt"
        linked = source / f"{split}_w_md5s.txt"
        rows, timestamps, split_articles = 0, set(), set()
        with path.open() as f, contextual.open() as fc, linked.open() as fl:
            while True:
                a, b, c = f.readline(), fc.readline(), fl.readline()
                if not a:
                    assert not b and not c, "Unequal row counts"
                    break
                original = tuple(map(int, a.split()))
                context_row = tuple(map(int, b.split()))
                fields = c.rstrip("\r\n").split("\t")
                assert original == context_row[:4] == tuple(map(int, fields[:4])), (
                    region,
                    split,
                    rows,
                )
                s, rel, o, t = original
                assert (
                    0 <= s < n
                    and 0 <= o < n
                    and 0 <= rel < r
                    and 0 <= context_row[4] < 5
                )
                timestamps.add(t)
                split_articles.update(fields[4].split(", "))
                if split == "train":
                    frequencies.update([s, o])
                rows += 1
        articles[split] = split_articles
        dump(out / f"{split}_article_md5s.json", sorted(split_articles))
        summary["splits"][split] = {
            "rows": rows,
            "t_min": min(timestamps),
            "t_max": max(timestamps),
            "days": len(timestamps),
            "date_min": dates[min(timestamps)],
            "date_max": dates[max(timestamps)],
            "articles": len(split_articles),
        }
        for p in [path, contextual, linked]:
            summary["source_sha256"]["data/SeCoGD/" + str(p.relative_to(base))] = (
                sha256(p)
            )
        split_ids.extend(
            {"split": split, "timestamp": t, "date": dates[t]}
            for t in sorted(timestamps)
        )
    assert summary["splits"]["train"]["t_max"] < summary["splits"]["valid"]["t_min"]
    assert summary["splits"]["valid"]["t_max"] < summary["splits"]["test"]["t_min"]
    fit_ids = set(json.loads((source / "train_md5_list.json").read_text()))
    all_ids = json.loads((source / "md5_list.json").read_text())
    assert fit_ids == articles["train"], (
        "Released fitting identifiers do not equal training-linked articles"
    )
    assert len(all_ids) == len(set(all_ids)) and fit_ids <= set(all_ids)
    dump(out / "lda_fit_article_md5s.json", sorted(fit_ids))
    threshold, cumulative, derived = sum(frequencies.values()) // 3, 0, []
    # Include the next entity while the previous cumulative count is at most the threshold.
    for entity, count in sorted(
        frequencies.items(), key=lambda item: item[1], reverse=True
    ):
        if cumulative <= threshold:
            derived.append(entity)
            cumulative += count
        else:
            break
    released = json.loads((source / "head_ents.json").read_text())
    assert set(derived) == set(released), (region, derived, released)
    summary["popularity"] = {
        "implementation": "subject+object occurrences in training fact rows; self-edge counts twice; descending frequency; add while previous cumulative <= floor(total/3)",
        "threshold_occurrences": threshold,
        "selected_occurrences": cumulative,
        "training_occurrences": sum(frequencies.values()),
        "excluded_targets": released,
        "candidate_entities_removed": False,
    }
    summary["articles"] = {
        "fit": len(fit_ids),
        "all": len(all_ids),
        "valid_also_train": len(articles["valid"] & fit_ids),
        "test_also_train": len(articles["test"] & fit_ids),
        "fit_scope": "unique article MD5s linked to training events",
        "validation_test_assignment": "frozen training dictionary and LDA inference; event label is argmax of mean topic probabilities across linked articles",
        "doc2vec_kmeans_gmm": "train-only doc2vec representation and clustering; frozen heldout assignment",
    }
    reference = json.loads((PROJECT_ROOT / "splits" / f"{region}.json").read_text())
    for key in [
        "entities",
        "relations",
        "days",
        "splits",
        "source_sha256",
        "popularity",
    ]:
        if summary[key] != reference[key]:
            raise ValueError(
                f"Released {region} data differs from the bundled split reference: {key}"
            )
    summary["input_fingerprint"] = manifest_fingerprint(summary)
    with (out / "split_dates.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "timestamp", "date"])
        writer.writeheader()
        writer.writerows(split_ids)
    dump(out / "dates.json", dates)
    dump(out / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument(
        "--regions", nargs="+", choices=["EG", "IR", "IS"], default=["EG", "IR", "IS"]
    )
    args = parser.parse_args()
    for name in ["data.zip", "data_disentangled.zip"]:
        path = args.data_root / name
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = args.data_root / info.filename
                if not target.resolve().is_relative_to(args.data_root.resolve()):
                    raise ValueError("Unsafe archive entry")
                if not target.exists() or (
                    not info.is_dir() and target.stat().st_size != info.file_size
                ):
                    archive.extract(info, args.data_root)
    summaries = [
        prepare_region(region, args.data_root, args.prepared_root)
        for region in args.regions
    ]
    dump(args.prepared_root / "data_manifest.json", summaries)


if __name__ == "__main__":
    main()
