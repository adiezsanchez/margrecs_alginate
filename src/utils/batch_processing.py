import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tifffile

from utils.feature_extraction import extract_nuclei_features_per_marker
from utils.io import (
    calculate_rescale_factor,
    ensure_output_dir,
    explore_lif_container,
    list_containers,
    load_lif_image,
    load_precomputed_results_if_available,
)
from utils.segmentation import predict_nuclei_labels_cp, predict_nuclei_labels_vo


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip()
    return sanitized if sanitized else "unnamed"


def log_step(
    level: str,
    container_id: str,
    image_name: str,
    stage: str,
    msg: str,
    elapsed_s: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_text = "na" if elapsed_s is None else f"{elapsed_s:.2f}"
    details_text = "na" if details is None else json.dumps(details, default=str)
    print(
        f"[{timestamp}] | {level} | container={container_id} | image={image_name} "
        f"| stage={stage} | msg={msg} | elapsed_s={elapsed_text} | details={details_text}"
    )


def save_image_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _as_tuple_int(values: Any, field_name: str, expected_len: int) -> tuple[int, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        raise ValueError(f"'{field_name}' must be a list/tuple with length {expected_len}, or null.")
    try:
        return tuple(int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain integer values.") from exc


def _as_optional_int_list(values: Any, field_name: str) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"'{field_name}' must be a list of integers or null.")
    try:
        return [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain integer values.") from exc


def _parse_markers(markers_raw: list[Any]) -> tuple[tuple[str, int], ...]:
    if not isinstance(markers_raw, list) or len(markers_raw) == 0:
        raise ValueError("'markers' must be a non-empty list.")

    parsed_markers: list[tuple[str, int]] = []
    for marker in markers_raw:
        if isinstance(marker, dict):
            name = str(marker.get("name", "")).strip()
            channel_raw = marker.get("channel")
        elif isinstance(marker, (list, tuple)) and len(marker) >= 2:
            name = str(marker[0]).strip()
            channel_raw = marker[1]
        else:
            raise ValueError("Each marker must be a mapping or a (name, channel) pair.")

        if not name:
            raise ValueError("Each marker requires a non-empty name.")

        try:
            channel = int(channel_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each marker requires integer 'channel'.") from exc

        parsed_markers.append((name, channel))

    return tuple(parsed_markers)


def _parse_nuclei_labeling(value: Any) -> str:
    labeling = str(value).strip().lower()
    if labeling not in {"cellpose", "voronoi-otsu"}:
        raise ValueError("'nuclei_labeling' must be 'cellpose' or 'voronoi-otsu'.")
    return labeling


def validate_runtime_config(config: dict[str, Any]) -> None:
    required_keys = {
        "raw_data_directory",
        "results_root",
        "markers",
        "nuclei_channel",
        "nuclei_labeling",
        "spot_sigma",
        "outline_sigma",
        "min_max_nuclei_volume",
        "container_indices",
        "image_indices",
        "overwrite_csv",
    }
    missing = sorted(required_keys.difference(config.keys()))
    if missing:
        raise ValueError(f"Missing runtime config keys: {missing}")


def _resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def build_runtime_config(
    user_config: dict[str, Any],
    raw_data_directory: str | Path,
    config_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    base_dir = Path(config_base_dir).resolve() if config_base_dir is not None else None
    config = {
        "raw_data_directory": _resolve_path(raw_data_directory, base_dir=base_dir),
        "results_root": _resolve_path(user_config.get("results_root", "./results"), base_dir=base_dir),
        "markers": _parse_markers(user_config["markers"]),
        "nuclei_channel": int(user_config["nuclei_channel"]),
        "nuclei_labeling": _parse_nuclei_labeling(user_config.get("nuclei_labeling", "cellpose")),
        "spot_sigma": float(user_config.get("spot_sigma", 2.0)),
        "outline_sigma": float(user_config.get("outline_sigma", 2.0)),
        "min_max_nuclei_volume": _as_tuple_int(
            user_config.get("min_max_nuclei_volume"),
            "min_max_nuclei_volume",
            expected_len=2,
        ),
        "container_indices": _as_optional_int_list(user_config.get("container_indices"), "container_indices"),
        "image_indices": _as_optional_int_list(user_config.get("image_indices"), "image_indices"),
        "overwrite_csv": bool(user_config.get("overwrite_csv", False)),
    }
    validate_runtime_config(config)
    return config


def process_single_image(
    lif_path: str,
    image_index: int,
    lif_container_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    image_start = time.perf_counter()
    image_name = f"image_{image_index}"

    try:
        log_step("INFO", lif_container_id, image_name, "image_start", "starting image processing", elapsed_s=0.0)

        stage_t0 = time.perf_counter()
        lif_image, lif_image_name, xml_metadata = load_lif_image(lif_path, image_index)
        image_name = lif_image_name
        safe_image_name = sanitize_filename(lif_image_name)
        csv_path = config["results_root"] / lif_container_id / f"{safe_image_name}.csv"
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "load",
            "image and metadata loaded",
            elapsed_s=time.perf_counter() - image_start,
            details={"stage_s": round(time.perf_counter() - stage_t0, 2), "image_index": image_index},
        )

        if csv_path.exists() and not config["overwrite_csv"]:
            log_step(
                "WARN",
                lif_container_id,
                image_name,
                "skip",
                "csv exists and overwrite disabled",
                elapsed_s=time.perf_counter() - image_start,
                details={"csv": str(csv_path)},
            )
            return {"status": "skipped", "lif_image_name": lif_image_name, "csv_path": csv_path}

        descriptor_dict = {
            "lif_container_id": lif_container_id,
            "lif_image_name": lif_image_name,
        }

        nuclei_labels_dir = ensure_output_dir(
            str(config["raw_data_directory"]),
            lif_container_id,
            results_type="nuclei_labels",
        )

        stage_t0 = time.perf_counter()
        nuclei_labels = load_precomputed_results_if_available(
            nuclei_labels_dir,
            safe_image_name,
            results_type="nuclei_labels",
        )

        if nuclei_labels is None:
            rescale_factor = calculate_rescale_factor(xml_metadata)
            if config["nuclei_labeling"] == "voronoi-otsu":
                nuclei_labels = predict_nuclei_labels_vo(
                    image=lif_image,
                    rescale_factor=rescale_factor,
                    nuclei_channel=config["nuclei_channel"],
                    min_max_nuclei_volume=config["min_max_nuclei_volume"],
                    spot_sigma=config["spot_sigma"],
                    outline_sigma=config["outline_sigma"],
                    visualize=False,
                )
            else:
                nuclei_labels = predict_nuclei_labels_cp(
                    image=lif_image,
                    rescale_factor=rescale_factor,
                    nuclei_channel=config["nuclei_channel"],
                    min_max_nuclei_volume=config["min_max_nuclei_volume"],
                    visualize=False,
                )
            nuclei_labels_path = nuclei_labels_dir / f"{safe_image_name}_nuclei_labels.tif"
            tifffile.imwrite(nuclei_labels_path, nuclei_labels)
            nuclei_source = "computed"
        else:
            nuclei_labels_path = nuclei_labels_dir / f"{safe_image_name}_nuclei_labels.tif"
            nuclei_source = "precomputed"

        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "nuclei",
            "nuclei labels ready",
            elapsed_s=time.perf_counter() - image_start,
            details={
                "stage_s": round(time.perf_counter() - stage_t0, 2),
                "source": nuclei_source,
                "n_labels": int(nuclei_labels.max()),
                "nuclei_labels_tif": str(nuclei_labels_path),
            },
        )

        stage_t0 = time.perf_counter()
        props_df = extract_nuclei_features_per_marker(
            nuclei_labels=nuclei_labels,
            lif_image=lif_image,
            markers=config["markers"],
            descriptor_dict=descriptor_dict,
        )
        props_df["input_img_shape"] = json.dumps([int(x) for x in nuclei_labels.shape])
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "features",
            "feature extraction complete",
            elapsed_s=time.perf_counter() - image_start,
            details={"stage_s": round(time.perf_counter() - stage_t0, 2), "n_rows": int(len(props_df))},
        )

        stage_t0 = time.perf_counter()
        save_image_csv(props_df, csv_path)
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "write_csv",
            "csv written",
            elapsed_s=time.perf_counter() - image_start,
            details={"stage_s": round(time.perf_counter() - stage_t0, 2), "csv": str(csv_path)},
        )

        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "done",
            "image processing completed",
            elapsed_s=time.perf_counter() - image_start,
            details={"n_rows": int(len(props_df))},
        )

        return {
            "status": "success",
            "lif_image_name": lif_image_name,
            "csv_path": csv_path,
            "n_rows": int(len(props_df)),
        }

    except Exception as exc:
        log_step(
            "ERROR",
            lif_container_id,
            image_name,
            "failed",
            "processing error",
            elapsed_s=time.perf_counter() - image_start,
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        print(traceback.format_exc())
        return {
            "status": "failed",
            "lif_image_name": image_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_batch(config: dict[str, Any]) -> dict[str, Any]:
    validate_runtime_config(config)
    batch_start = time.perf_counter()
    config["results_root"].mkdir(parents=True, exist_ok=True)

    lif_containers = list_containers(str(config["raw_data_directory"]), file_format="lif")
    if not lif_containers:
        raise FileNotFoundError(f"No .lif containers found in '{config['raw_data_directory']}'.")

    if config["container_indices"] is None:
        selected_containers = lif_containers
    else:
        selected_containers = [lif_containers[i] for i in config["container_indices"]]

    processed = 0
    success = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []

    for lif_path in selected_containers:
        nr_imgs, lif_container_id = explore_lif_container(lif_path, display=False)

        if config["image_indices"] is None:
            image_indices = list(range(nr_imgs))
        else:
            image_indices = [i for i in config["image_indices"] if i < nr_imgs]

        log_step(
            "INFO",
            lif_container_id,
            "all",
            "container_start",
            "starting container batch",
            details={"images_selected": len(image_indices), "images_total": nr_imgs},
        )

        for image_index in image_indices:
            result = process_single_image(
                lif_path=lif_path,
                image_index=image_index,
                lif_container_id=lif_container_id,
                config=config,
            )
            processed += 1
            status = result["status"]
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(
                    {
                        "container": lif_container_id,
                        "image": str(result.get("lif_image_name", f"image_{image_index}")),
                        "error_type": str(result.get("error_type", "UnknownError")),
                        "error": str(result.get("error", "Unknown error")),
                    }
                )

    elapsed_min = (time.perf_counter() - batch_start) / 60.0
    print(
        "BATCH_SUMMARY"
        f" | containers={len(selected_containers)}"
        f" | images_total={processed}"
        f" | processed={processed}"
        f" | success={success}"
        f" | skipped={skipped}"
        f" | failed={failed}"
        f" | elapsed_min={elapsed_min:.2f}"
    )

    if failures:
        for item in failures:
            print(
                "FAILED_ITEM"
                f" | container={item['container']}"
                f" | image={item['image']}"
                f" | error_type={item['error_type']}"
                f" | error={item['error']}"
            )

    return {
        "containers": len(selected_containers),
        "images_total": processed,
        "processed": processed,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "elapsed_min": elapsed_min,
        "failures": failures,
    }
