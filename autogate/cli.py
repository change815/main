"""Command line interface for autogate."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from .config import load_config, summarize_config, validate_config
from .eval_f1 import evaluate
from .pipeline import AutoGatePipeline, configure_logging

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

    predictions_value = args.predictions or config.get("io", {}).get("out_dir", "./out/gates")
    predictions_dir = Path(predictions_value)
    truth_value = args.truth or config.get("io", {}).get("eval_truth_csv")
    if not truth_value:
        raise FileNotFoundError("Truth CSV must be provided via --truth or io.eval_truth_csv")
    truth_csv = Path(truth_value)
    if not truth_csv.exists():
        raise FileNotFoundError(f"Truth CSV not found: {truth_csv}")

    results = evaluate(config, predictions_dir, truth_csv)
    LOGGER.info("Evaluation results: %s", json.dumps({k: v for k, v in results.items() if k != "metrics"}, indent=2))
    print(f"Evaluation complete. Logs: {log_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elastic image registration-based auto-gating tool")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init-config", help="Generate example configuration")
    init_parser.add_argument("--directory", default="./cfg", help="Directory to place example configuration")
    init_parser.set_defaults(func=cmd_init_config)

    run_parser = subparsers.add_parser("run", help="Execute end-to-end auto-gating")
    _add_common_run_arguments(run_parser)
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser("eval", help="Evaluate predictions against truth CSV")
    _add_common_run_arguments(eval_parser)
    eval_parser.add_argument("--predictions", help="Directory containing predicted gate CSVs", default=None)
    eval_parser.add_argument("--truth", help="Path to truth gates CSV", default=None)
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
