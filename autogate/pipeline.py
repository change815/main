"""End-to-end pipeline implementation."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


from .config import ensure_ranges
from .gates import (
    Gate,
    assign_populations,
    read_gates,
    transform_gate,
    write_gates,
)
from .imaging import DensityImage, compute_density_image
from .io_fcs import FCSLoader
from .register import RegistrationError, register_bspline
from .selector import select_best
from .visualize import plot_population_scatter

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".fcs", ".csv", ".npz", ".npy"}


def configure_logging(log_dir: Path, level: str) -> Path:
    """Configure logging to console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"autogate-{timestamp}.log"

    logging_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    LOGGER.info("Logging configured. File: %s", log_path)
    return log_path


class AutoGatePipeline:
    """Full auto-gating pipeline."""

    def __init__(self, config: Mapping[str, any]) -> None:
        self.config = config
        panel_cfg = config.get("panel", {})
        imaging_cfg = config.get("imaging", {})

        channels = panel_cfg.get("channels", [])
        ensure_ranges(panel_cfg, channels)

        self.loader = FCSLoader(
            channels=channels,
            transform=panel_cfg.get("transform", "asinh"),
            compensation=panel_cfg.get("compensation", "auto"),
            ranges=panel_cfg.get("ranges", {}),
        )
        self.bins = int(imaging_cfg.get("bins", 256))
        self.smooth_sigma = float(imaging_cfg.get("smooth_sigma", 0.0))
        self.training_density: Dict[str, Dict[Tuple[str, str], DensityImage]] = defaultdict(dict)
        self.training_gates: Dict[str, Dict[Tuple[str, str], List[Gate]]] = defaultdict(lambda: defaultdict(list))
        self.training_data_paths: Dict[str, Path] = {}
        io_cfg = config.get("io", {})
        self.gates_out_dir = Path(io_cfg.get("out_dir", "./out/gates"))
        labels_default = self.gates_out_dir / "event_labels"
        plots_default = self.gates_out_dir / "plots"
        self.labels_dir = Path(io_cfg.get("labels_dir", labels_default))
        self.plots_dir = Path(io_cfg.get("plots_dir", plots_default))

    def _list_fcs_files(self, directory: Path) -> List[Path]:
        if not directory.exists():
            raise FileNotFoundError(f"FCS directory not found: {directory}")
        files = [p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]
        if not files:
            raise FileNotFoundError(f"No supported files found in {directory}")
        return sorted(files)

    def prepare_training(self) -> None:
        io_cfg = self.config.get("io", {})
        train_dir = Path(io_cfg.get("train_fcs_dir"))
        train_gates_csv = Path(io_cfg.get("train_gates_csv"))
        gate_defs = read_gates(train_gates_csv)
        training_paths = self._list_fcs_files(train_dir)
        training_ids = [p.stem for p in training_paths]

        for path in training_paths:
            LOGGER.info("Loading training data %s", path)
            data = self.loader.read_fcs(path)
            self.training_data_paths[path.stem] = path
            plots = self._plots_from_gates(gate_defs)
            for plot in plots:
                channel_x, channel_y = plot
                density = compute_density_image(
                    data,
                    channel_x,
                    channel_y,
                    bins=self.bins,
                    smooth_sigma=self.smooth_sigma,
                    ranges=self.loader.ranges,
                )
                self.training_density[path.stem][plot] = density

        default_training = training_ids[0]
        for gate in gate_defs:
            training_id = Path(gate.fcs_file).stem if gate.fcs_file else default_training
            if training_id not in training_ids:
                raise ValueError(
                    f"Gate {gate.gate_id} references unknown training file {gate.fcs_file}"
                )
            plot = (gate.channel_x, gate.channel_y)
            self.training_gates[training_id][plot].append(gate)

    def _plots_from_gates(self, gates: Iterable[Gate]) -> List[Tuple[str, str]]:
        plots = sorted({(gate.channel_x, gate.channel_y) for gate in gates})
        return plots

    def _prepare_target_density(self, data, plots: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], DensityImage]:
        densities = {}
        for plot in plots:
            channel_x, channel_y = plot
            densities[plot] = compute_density_image(
                data,
                channel_x,
                channel_y,
                bins=self.bins,
                smooth_sigma=self.smooth_sigma,
                ranges=self.loader.ranges,
            )
        return densities

    def run(self) -> Dict[str, any]:
        self.prepare_training()
        io_cfg = self.config.get("io", {})
        target_dir = Path(io_cfg.get("target_fcs_dir"))
        out_dir = self.gates_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        target_files = self._list_fcs_files(target_dir)
        plots = sorted({plot for per_train in self.training_gates.values() for plot in per_train.keys()})
        summary = {
            "targets": [],
            "selections": [],
        }

        for target_path in target_files:
            start = time.time()
            LOGGER.info("Processing target file %s", target_path.name)
            data = self.loader.read_fcs(target_path)
            target_densities = self._prepare_target_density(data, plots)
            all_transformed: List[Gate] = []
            selection_records = []

            for plot in plots:
                training_candidates = {
                    train_id: self.training_density[train_id][plot].image
                    for train_id in self.training_density
                    if plot in self.training_density[train_id]
                }
                if not training_candidates:
                    LOGGER.warning("No training densities available for plot %s", plot)
                    continue
                target_image = target_densities[plot].image
                try:
                    best_id, distance = select_best(training_candidates, target_image)
                except ValueError as exc:
                    LOGGER.warning("Selection failed for plot %s: %s", plot, exc)
                    continue
                selection_records.append(
                    {
                        "target": target_path.name,
                        "plot": f"{plot[0]}__{plot[1]}",
                        "training_id": best_id,
                        "distance": distance,
                    }
                )
                training_density = self.training_density[best_id][plot]
                reg_cfg = self.config.get("registration", {})
                try:
                    transform = register_bspline(
                        fixed=target_image,
                        moving=training_density.image,
                        grid_spacing=int(reg_cfg.get("grid_spacing", 32)),
                        levels=int(reg_cfg.get("levels", 3)),
                        iterations=int(reg_cfg.get("iterations", 200)),
                    )
                except RegistrationError as exc:
                    LOGGER.warning(
                        "Registration failed for plot %s using %s: %s. Falling back to identity.",
                        plot,
                        best_id,
                        exc,
                    )
                    transform = register_bspline_identity()

                for gate in self.training_gates[best_id][plot]:
                    transformed = transform_gate(
                        gate,
                        transform,
                        training_density,
                        target_densities[plot],
                    )
                    all_transformed.append(transformed)

            out_path = out_dir / f"{target_path.stem}_gates.csv"
            write_gates(out_path, all_transformed)

            labels = assign_populations(data, all_transformed)
            labels_path = self.labels_dir / f"{target_path.stem}_labels.csv"
            labels.to_csv(labels_path, index=False)
            LOGGER.info(
                "Assigned populations for %s: %d labeled events (of %d)",
                target_path.name,
                int((labels["population"] != "UNGATED").sum()),
                len(labels),
            )

            for plot in plots:
                channel_x, channel_y = plot
                figure_path = self.plots_dir / f"{target_path.stem}__{channel_x}__{channel_y}.png"
                try:
                    plot_population_scatter(
                        data,
                        labels["population"],
                        channel_x,
                        channel_y,
                        figure_path,
                    )
                except Exception as exc:  # pragma: no cover - plotting errors are rare
                    LOGGER.warning(
                        "Failed to render plot for %s on plot %s/%s: %s",
                        target_path.name,
                        channel_x,
                        channel_y,
                        exc,
                    )

            duration = time.time() - start
            LOGGER.info(
                "Completed %s in %.2f seconds with %d gates", target_path.name, duration, len(all_transformed)
            )
            summary["targets"].append({
                "file": target_path.name,
                "gates": len(all_transformed),
                "duration": duration,
                "gate_file": str(out_path),
                "labels_file": str(labels_path),
            })
            summary["selections"].extend(selection_records)
        return summary


def register_bspline_identity():
    import SimpleITK as sitk

    transform = sitk.Transform(2, sitk.sitkIdentity)
    return transform
