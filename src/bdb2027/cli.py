"""Command line entry points for the research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from bdb2027.config import Project


def _project(args) -> Project:
    data_root = Path(args.data_root) if args.data_root else None
    config = Path(args.config) if args.config else None
    return Project(config_path=config, data_root=data_root)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="bdb2027", description="The Cost of a Cut research pipeline")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="Write the data audit, coverage, and feasibility report")
    audit.set_defaults(func=_audit)

    extract = sub.add_parser("extract", help="Detect cuts for development cohorts")
    extract.add_argument("--cohorts", nargs="+", type=int, required=True)
    extract.set_defaults(func=_extract)

    gallery = sub.add_parser("review-gallery", help="Outcome-blind trajectory plots for manual review")
    gallery.set_defaults(func=_gallery)

    protocol = sub.add_parser("protocol", help="Write reports/protocol.md from the locked configuration")
    protocol.set_defaults(func=_protocol)

    develop = sub.add_parser("develop", help="2023 grouped validation and the 2024 chronological check")
    develop.set_defaults(func=_develop)

    freeze = sub.add_parser("freeze", help="Refit 2023-24 and write the freeze manifest")
    freeze.set_defaults(func=_freeze)

    evaluate = sub.add_parser("evaluate-heldout", help="One-shot evaluation of a held-out draft class")
    evaluate.add_argument("--cohort", type=int, required=True)
    evaluate.set_defaults(func=_evaluate)

    report = sub.add_parser("report", help="Figures, metrics, claim ledger, and the result summary")
    report.set_defaults(func=_report)

    smoke = sub.add_parser("smoke", help="Labeled smoke run on a deterministic 2023 subset")
    smoke.set_defaults(func=_smoke)

    args = parser.parse_args(argv)
    args.func(args)


def _audit(args):
    from bdb2027.audit import run_audit

    run_audit(_project(args))


def _extract(args):
    from bdb2027.extract import extract

    extract(_project(args), cohorts=list(args.cohorts), allow_outcomes=False)


def _gallery(args):
    from bdb2027.gallery import write_gallery

    write_gallery(_project(args))


def _protocol(args):
    from bdb2027.pipeline import write_protocol

    write_protocol(_project(args))


def _develop(args):
    from bdb2027.pipeline import develop

    develop(_project(args))


def _freeze(args):
    from bdb2027.pipeline import freeze

    freeze(_project(args))


def _evaluate(args):
    from bdb2027.pipeline import evaluate_heldout

    evaluate_heldout(_project(args), cohort=int(args.cohort))


def _report(args):
    from bdb2027.pipeline import report

    report(_project(args))


def _smoke(args):
    from bdb2027.pipeline import smoke

    smoke(_project(args))


if __name__ == "__main__":
    main()
