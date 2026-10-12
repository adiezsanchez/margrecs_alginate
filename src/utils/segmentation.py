from cellpose import models, core, io
import numpy as np
import pyclesperanto_prototype as cle
from skimage.filters import difference_of_gaussians
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.morphology import binary_closing, binary_opening, binary_erosion, disk, ball
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from skimage.transform import resize

io.logger_setup()  # run this to get printing of progress

_CELLPOSE_MODEL = None


def _get_cellpose_model(require_gpu: bool = True):
    """
    Lazily initialize and cache the Cellpose model.

    This keeps module import lightweight and allows workflows that only load
    precomputed results to run on CPU-only environments.
    """
    global _CELLPOSE_MODEL

    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL

    has_gpu = core.use_gpu()
    if require_gpu and not has_gpu:
        raise RuntimeError(
            "Cellpose nuclei prediction requires GPU, but no GPU was detected. "
            "You can still run workflows that load precomputed nuclei labels."
        )

    _CELLPOSE_MODEL = models.CellposeModel(gpu=has_gpu)
    return _CELLPOSE_MODEL


def _resolve_napari_viewer(viewer):
    """
    Return a napari Viewer for optional visualization.

    If ``viewer`` is passed, it is used. Otherwise tries ``napari.current_viewer()``;
    if none exists, creates ``napari.Viewer()``. Import is deferred until visualization runs.
    """
    if viewer is not None:
        return viewer
    import napari

    v = napari.current_viewer()
    if v is not None:
        return v
    return napari.Viewer()


def _keep_objects_in_size_range(labels: np.ndarray, min_max_size: tuple[int, int] | None) -> np.ndarray:
    """
    Keep only labeled objects whose voxel count is within a min/max range.

    Args:
        labels (np.ndarray): Labeled image where 0 is background.
        min_max_size (tuple[int, int] | None): Inclusive size range as (min_size, max_size).
            If None, labels are returned unchanged (no size filtering).

    Returns:
        np.ndarray: Filtered labels, relabeled sequentially from 1..N.
    """
    if min_max_size is None:
        return labels

    min_size, max_size = min_max_size
    counts = np.bincount(labels.ravel())
    keep = (counts >= max(min_size, 0)) & (counts <= max_size)
    keep[0] = False  # keep background as 0

    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    filtered, _, _ = relabel_sequential(filtered)
    return filtered

def predict_nuclei_labels_cp(image: np.ndarray, rescale_factor: float, nuclei_channel: int, min_max_nuclei_volume: tuple[int, int] | None = None, visualize=False, viewer=None) -> np.ndarray:
    """
    Predict nuclei labels using CellposeSAM using anisotropy correction.

    Args:
        image (np.ndarray): Image to predict nuclei labels from.
        rescale_factor (float): Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
        nuclei_channel (int): Channel index of the nuclei channel in the image.
        min_max_nuclei_volume (tuple[int, int] | None, optional): Inclusive min/max nuclei volume
            used to filter predicted labels. Defaults to None (no volume filtering).
        visualize (bool, optional): If True, display the predicted nuclei labels in Napari.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.

    Returns:
        np.ndarray: Nuclei labels.
    """
    model = _get_cellpose_model(require_gpu=True)

    # Predict nuclei labels
    nuclei_labels, _ , _ = model.eval(image[nuclei_channel], do_3D=True, anisotropy=rescale_factor, z_axis=0, niter=1000)
    # Filter nuclei labels to keep only those within the specified size range
    nuclei_labels = _keep_objects_in_size_range(nuclei_labels, min_max_nuclei_volume)

    # Display the resulting nuclei labels in Napari if requested.
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_labels(nuclei_labels)

    return nuclei_labels


def predict_nuclei_labels_vo(
    image: np.ndarray,
    rescale_factor: float,
    nuclei_channel: int,
    min_max_nuclei_volume: tuple[int, int] | None = None,
    spot_sigma: float = 5,
    outline_sigma: float = 1,
    visualize=False,
    viewer=None,
) -> np.ndarray:
    """
    Predict nuclei labels using pyclesperanto Voronoi-Otsu labeling with anisotropy correction.

    Args:
        image (np.ndarray): Image to predict nuclei labels from.
        rescale_factor (float): Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
        nuclei_channel (int): Channel index of the nuclei channel in the image.
        min_max_nuclei_volume (tuple[int, int] | None, optional): Inclusive min/max nuclei volume
            used to filter predicted labels. Defaults to None (no volume filtering).
        visualize (bool, optional): If True, display the predicted nuclei labels in Napari.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.
        spot_sigma (float, optional): Controls how close detected nuclei can be in Voronoi-Otsu
            labeling. Larger values merge nearby spots. Defaults to 5.
        outline_sigma (float, optional): Controls how precise segmented objects are outlined.
            Larger values produce smoother boundaries. Defaults to 1.

    Returns:
        np.ndarray: Nuclei labels.
    """
    nuclei_channel_image = image[nuclei_channel].astype(np.float32)

    # Voronoi-Otsu assumes isotropic voxels: stretch Z by the anisotropy rescale factor
    nuclei_iso = cle.scale(
        nuclei_channel_image,
        factor_x=1,
        factor_y=1,
        factor_z=float(rescale_factor),
        auto_size=True,
        linear_interpolation=True,
    )

    # Predict nuclei labels
    labels_iso = cle.voronoi_otsu_labeling(
        nuclei_iso,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
    )

    # Map labels back to the original anisotropic grid (nearest-neighbor)
    labels_iso_np = np.asarray(cle.pull(labels_iso))
    nuclei_labels = resize(
        labels_iso_np,
        nuclei_channel_image.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint32)

    # Filter nuclei labels to keep only those within the specified size range
    nuclei_labels = _keep_objects_in_size_range(nuclei_labels, min_max_nuclei_volume)

    # Display the resulting nuclei labels in Napari if requested.
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_labels(nuclei_labels)

    return nuclei_labels


def clean_colocalized_labels(
    labels_to_clean: np.ndarray,
    colocalizing_labels: np.ndarray,
) -> np.ndarray:
    """
    Remove voxels from ``labels_to_clean`` that overlap with ``colocalizing_labels``.

    Intended for imperfect channel colocalization (e.g. zero HPFa_red voxels that
    overlap A549_green predictions). Label IDs are preserved so downstream
    comparisons by label (e.g. percent area removed) remain valid.

    Args:
        labels_to_clean (np.ndarray): Label image to clean (e.g. HPFa_red).
        colocalizing_labels (np.ndarray): Label image used as a mask (e.g. A549_green).
            Any voxel where this is > 0 is zeroed in ``labels_to_clean``.

    Returns:
        np.ndarray: Cleaned labels (copy); originals are left unchanged.
    """
    if labels_to_clean.shape != colocalizing_labels.shape:
        raise ValueError(
            "labels_to_clean and colocalizing_labels must have the same shape, "
            f"got {labels_to_clean.shape} and {colocalizing_labels.shape}."
        )

    cleaned = labels_to_clean.copy()
    cleaned[colocalizing_labels > 0] = 0
    return cleaned
