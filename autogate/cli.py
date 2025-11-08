"""Command line interface for autogate."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config, summarize_config, validate_config
from .eval_f1 import evaluate
from .pipeline import AutoGatePipeline, configure_logging
from .prepare import build_gates_from_annotations, write_summary

LOGGER = logging.getLogger(__name__)


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="./cfg/panel.yaml", help="Path to YAML configuration file")
    parser.add_argument("--bins", type=int, help="Override imaging bins", default=None)
    parser.add_argument("--out-dir", type=str, help="Override output directory", default=None)


def cmd_init_config(args: argparse.Namespace) -> None:
    cfg_dir = Path(args.directory)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    example_path = Path(__file__).resolve().parent.parent / "cfg" / "panel.example.yaml"
    target_path = cfg_dir / "panel.example.yaml"
    target_path.write_text(example_path.read_text())
    print(f"Example configuration written to {target_path}")


def cmd_prepare_annotations(args: argparse.Namespace) -> None:
    summary = build_gates_from_annotations(
        csv_root=args.csv_root,
        spec_path=args.spec,
        output_csv=args.output,
        fcs_suffix=args.fcs_suffix,
        expand_frac=args.expand,
        min_points=args.min_points,
    )
    LOGGER.info("Annotation summary: %s", json.dumps(summary, indent=2))
    if args.summary:
        write_summary(summary, args.summary)
    print(f"Generated gates CSV at {summary['gates_csv']}")


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.bins is not None:
        config.setdefault("imaging", {})["bins"] = args.bins
    if args.out_dir is not None:
        config.setdefault("io", {})["out_dir"] = args.out_dir

    validate_config(config)
    log_dir = Path(config.get("logging", {}).get("log_dir", "./out/logs"))
    log_file = configure_logging(log_dir, config.get("logging", {}).get("level", "INFO"))
    LOGGER.info("Using configuration: %s", json.dumps(summarize_config(config), indent=2))

    pipeline = AutoGatePipeline(config)
    summary = pipeline.run()
    LOGGER.info("Run summary: %s", json.dumps(summary, indent=2))
    print(f"Completed auto-gating. Logs: {log_file}")


def cmd_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validate_config(config)
    log_dir = Path(config.get("logging", {}).get("log_dir", "./out/logs"))
    log_file = configure_logging(log_dir, config.get("logging", {}).get("level", "INFO"))

    eval_cfg = config.get("evaluation", {})
    mode = (args.mode or eval_cfg.get("mode", "auto")).lower()
    default_predictions = config.get("io", {}).get("out_dir", "./out/gates")
    if mode in {"auto", "labels"}:
        default_predictions = config.get("io", {}).get(
            "labels_dir",
            Path(default_predictions) / "event_labels",
        )
    predictions_value = args.predictions or default_predictions
    predictions_dir = Path(predictions_value)
    truth_value = args.truth or config.get("io", {}).get("eval_truth_csv")
    if not truth_value:
        raise FileNotFoundError("Truth CSV must be provided via --truth or io.eval_truth_csv")
    truth_csv = Path(truth_value)
    if not truth_csv.exists():
        raise FileNotFoundError(f"Truth CSV not found: {truth_csv}")

    results = evaluate(config, predictions_dir, truth_csv, mode=mode)
    LOGGER.info("Evaluation results: %s", json.dumps({k: v for k, v in results.items() if k != "metrics"}, indent=2))
    print(f"Evaluation complete. Logs: {log_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elastic image registration-based auto-gating tool")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init-config", help="Generate example configuration")
    init_parser.add_argument("--directory", default="./cfg", help="Directory to place example configuration")
    init_parser.set_defaults(func=cmd_init_config)

    prepare_parser = subparsers.add_parser(
        "prepare-annotations",
        help="Convert annotated CSV directories into training gates and summary metadata",
    )
    prepare_parser.add_argument("--csv-root", required=True, help="Root directory containing per-sample annotation subfolders")
    prepare_parser.add_argument("--spec", required=True, help="YAML specification describing coordinate systems and populations")
    prepare_parser.add_argument("--output", default="./data/train/gates.csv", help="Path to write generated gates CSV")
    prepare_parser.add_argument("--summary", default=None, help="Optional YAML path to store summary info (channels/ranges)")
    prepare_parser.add_argument("--fcs-suffix", default=".fcs", help="Suffix to append to sample id for fcs_file column")
    prepare_parser.add_argument("--expand", type=float, default=0.05, help="Fractional expansion applied to convex hull polygons")
    prepare_parser.add_argument("--min-points", type=int, default=20, help="Warn when a population has fewer annotated points than this threshold")
    prepare_parser.set_defaults(func=cmd_prepare_annotations)

    run_parser = subparsers.add_parser("run", help="Execute end-to-end auto-gating")
    _add_common_run_arguments(run_parser)
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser("eval", help="Evaluate predictions against truth CSV")
    _add_common_run_arguments(eval_parser)
    eval_parser.add_argument("--predictions", help="Directory containing prediction CSVs", default=None)
    eval_parser.add_argument("--truth", help="Path to truth data (CSV or directory)", default=None)
    eval_parser.add_argument(
        "--mode",
        choices=["auto", "gates", "labels"],
        default=None,
        help="Evaluation mode (auto-detect, polygon gates, or per-event labels)",
    )
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
