"""Representative time series and descriptive activity-group associations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ha_afsnet.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
)
from ha_afsnet.config import (
    PROJECT_ROOT as ROOT,
)
from ha_afsnet.data import RegionalData
from ha_afsnet.statistics import write_csv
from ha_afsnet.utils import dump


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export five-seed gate time series and descriptive activity groups"
    )
    parser.add_argument(
        "--regions", nargs="+", choices=["EG", "IR", "IS"], default=["EG", "IR", "IS"]
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    out = args.output_root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    groups, combined = [], []
    for region in args.regions:
        files = sorted(
            (args.output_root / f"runs/HA-AFSNet/{region}").glob("seed_*/gates.csv")
        )
        if len(files) != 5 or not all(
            (path.parent / "result.json").exists() for path in files
        ):
            continue
        data = RegionalData(region, "cpu", args.data_root, args.prepared_root)
        quartiles = np.quantile(data.row_activity[data.splits["train"]], [0.25, 0.75])
        dump(
            out / f"{region}_activity_thresholds.json",
            {
                "scope": "all 2068 training dates",
                "activity": "log(1 + unique incident entities + fact rows)",
                "q25": quartiles[0],
                "q75": quartiles[1],
                "low": "<= q25",
                "high": ">= q75",
                "gate_date": "forecast t; activity uses t-1",
                "gate_input_activity": "Eq5 deduplicated edge union; diagnostic row activity only determines groups",
                "dates": "2015-02-19..2020-10-17 thresholds; gate groups only exported training forecasts with five increments",
                "interpretation": "descriptive associations; activity and composition are gate inputs",
            },
        )
        results = [
            json.loads((path.parent / "result.json").read_text()) for path in files
        ]
        assert all(
            result["status"] == "completed"
            and result.get("full_test")
            and not result.get("smoke")
            for result in results
        ), "Gate files must come from completed full training"
        assert (
            len({result["protocol_sha256"] for result in results}) == 1
            and len({result["data_manifest_sha256"] for result in results}) == 1
        ), "Only aggregate matched protocols/splits"
        frame = pd.concat(
            [pd.read_csv(path, dtype={"date": str}) for path in files],
            ignore_index=True,
        )
        assert sorted(frame.seed.unique()) == [42, 43, 44, 45, 46], (
            "Only aggregate after all five seeds"
        )
        assert not frame.duplicated(["seed", "timestamp", "context"]).any()
        assert np.isfinite(frame.alpha).all() and frame.alpha.between(0, 1).all()
        for result, path in zip(results, files):
            config = json.loads((path.parent / "config.json").read_text())
            current = frame[frame.seed == result["seed"]]
            expected_dates = list(range(config["history"] + 1, max(data.rows) + 1))
            assert (
                sorted(current.timestamp.unique()) == expected_dates
                and len(current) == len(expected_dates) * config["contexts"]
            ), "Incomplete gate dates/contexts"
        train = frame[frame["split"] == "train"].copy()
        train["activity_group"] = np.where(
            train.diagnostic_row_activity <= quartiles[0],
            "low",
            np.where(train.diagnostic_row_activity >= quartiles[1], "high", "middle"),
        )
        for (seed, k), current in train.groupby(["seed", "context"]):
            low = current.loc[current.activity_group == "low", "alpha"]
            high = current.loc[current.activity_group == "high", "alpha"]
            groups.append(
                {
                    "dataset": region,
                    "seed": seed,
                    "context": k,
                    "mean_alpha": current.alpha.mean(),
                    "temporal_sd": current.alpha.std(ddof=1),
                    "low_mean": low.mean(),
                    "high_mean": high.mean(),
                    "high_minus_low": high.mean() - low.mean(),
                    "low_dates": len(low),
                    "high_dates": len(high),
                }
            )
        daily = (
            frame.groupby(["dataset", "split", "timestamp", "date", "context"])
            .agg(
                alpha_mean=("alpha", "mean"),
                alpha_seed_sd=("alpha", "std"),
                n_seeds=("seed", "nunique"),
                activity=("activity", "first"),
                activity_change=("activity_change", "first"),
                composition_change=("composition_change", "first"),
            )
            .reset_index()
        )
        combined.append(daily)
        plot = daily[daily["split"] == "test"].copy()
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True, constrained_layout=True
        )
        for k, current in plot.groupby("context"):
            dates = pd.to_datetime(current["date"], format="%Y%m%d")
            y = current.alpha_mean.to_numpy()
            sd = current.alpha_seed_sd.to_numpy()
            axes[0].plot(dates, y, label=f"context {k}")
            axes[0].fill_between(dates, y - sd, y + sd, alpha=0.12)
        observed = plot.drop_duplicates("timestamp")
        dates = pd.to_datetime(observed["date"], format="%Y%m%d")
        axes[1].plot(dates, observed.activity)
        axes[2].plot(dates, observed.composition_change)
        axes[0].set_ylabel("Gate alpha")
        axes[0].legend(ncol=5)
        axes[0].set_ylim(0, 1)
        axes[1].set_ylabel("Activity (t-1)")
        axes[2].set_ylabel("Composition KL")
        axes[2].set_xlabel("Forecast date")
        fig.suptitle(f"{region}: five-seed mean gate, test dates; shading is seed SD")
        fig.savefig(figures / f"{region}_gate_timeseries.png", dpi=200)
        plt.close(fig)
    write_csv(
        out / "gate_activity_per_seed.csv",
        groups,
        [
            "dataset",
            "seed",
            "context",
            "mean_alpha",
            "temporal_sd",
            "low_mean",
            "high_mean",
            "high_minus_low",
            "low_dates",
            "high_dates",
        ],
    )
    summary = []
    if groups:
        for (region, k), current in pd.DataFrame(groups).groupby(
            ["dataset", "context"]
        ):
            for metric in [
                "mean_alpha",
                "temporal_sd",
                "low_mean",
                "high_mean",
                "high_minus_low",
            ]:
                summary.append(
                    {
                        "dataset": region,
                        "context": k,
                        "metric": metric,
                        "n_seeds": len(current),
                        "mean": float(current[metric].mean()),
                        "sample_sd": float(current[metric].std(ddof=1)),
                    }
                )
    write_csv(
        out / "gate_activity_summary.csv",
        summary,
        ["dataset", "context", "metric", "n_seeds", "mean", "sample_sd"],
    )
    if combined:
        pd.concat(combined).to_csv(out / "gate_timeseries.csv", index=False)
    else:
        write_csv(
            out / "gate_timeseries.csv",
            [],
            [
                "dataset",
                "split",
                "timestamp",
                "date",
                "context",
                "alpha_mean",
                "alpha_seed_sd",
                "n_seeds",
                "activity",
                "activity_change",
                "composition_change",
            ],
        )
    print(json.dumps({"regions_with_full_gate_exports": len(combined)}))


if __name__ == "__main__":
    main()
