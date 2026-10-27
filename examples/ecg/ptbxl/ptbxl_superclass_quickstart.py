"""PTB-XL 5-superclass multi-label quickstart.

Requires::

    pip install 'pyhealth[ptbxl]'

Download PTB-XL v1.0.3 from https://physionet.org/content/ptb-xl/1.0.3/
and point ``--root`` at the extracted version directory (contains
``ptbxl_database.csv``, ``scp_statements.csv``, ``records100/``).

Example::

    python examples/ecg/ptbxl/ptbxl_superclass_quickstart.py \\
        --root /data/ptb-xl/1.0.3 --dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyhealth.datasets import PTBXLDataset, split_by_strat_fold
from pyhealth.tasks import PTBXLSuperclassClassification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="PTB-XL version root (ptbxl_database.csv + records*/)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        choices=(100, 500),
        help="Waveform sampling rate (default: 100 Hz / filename_lr)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Limit patients via BaseDataset.dev mode",
    )
    args = parser.parse_args()

    dataset = PTBXLDataset(
        root=str(args.root),
        sampling_rate=args.sampling_rate,
        dev=args.dev,
    )
    task = PTBXLSuperclassClassification(
        scp_statements_path=str(args.root / "scp_statements.csv"),
    )
    samples = dataset.set_task(task)
    # Read strat_fold from ptbxl_database.csv and align it with sample indices
    # through record_to_index, so the split decodes no sample. Row order cannot
    # be used: set_task drops records with empty label sets.
    db = pd.read_csv(
        args.root / "ptbxl_database.csv", usecols=["ecg_id", "strat_fold"]
    )
    fold_by_record = {str(int(e)): int(f) for e, f in zip(db.ecg_id, db.strat_fold)}
    folds: list[int | None] = [None] * len(samples)
    for record_id, indices in samples.record_to_index.items():
        for i in indices:
            folds[i] = fold_by_record[str(record_id)]
    if None in folds:
        raise ValueError("some samples have no record_id in ptbxl_database.csv")
    train, val, test = split_by_strat_fold(samples, folds=folds)
    print(
        f"samples={len(samples)} "
        f"train={len(train)} val={len(val)} test={len(test)}"
    )
    if len(samples):
        first = samples[0]
        print(
            "first sample:",
            {
                "patient_id": first["patient_id"],
                "record_id": first["record_id"],
                "labels": first["labels"],
                "strat_fold": first["strat_fold"],
                "signal_shape": tuple(first["signal"].shape),
            },
        )


if __name__ == "__main__":
    main()
