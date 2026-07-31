import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MORPHOLOGY_PROPERTIES = [
    "label",
    "centroid",
    "area",
    "area_bbox",
    "area_filled",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "inertia_tensor_eigvals",
]

INTENSITY_PROPERTIES = [
    "label",
    "intensity_mean",
    "intensity_min",
    "intensity_max",
    "intensity_std",
]


def extract_nuclei_features_per_marker(
    nuclei_labels: np.ndarray,
    lif_image: np.ndarray,
    markers: list[tuple[str, int, str]],
    descriptor_dict: dict[str, str | int | float | bool],
) -> pd.DataFrame:
    """
    Extract morphology and per-marker intensity features for each nucleus label.

    The function computes a base morphology table from ``nuclei_labels`` using
    ``MORPHOLOGY_PROPERTIES`` and then appends per-channel intensity statistics for
    all markers except ``"brightfield"`` using ``INTENSITY_PROPERTIES``.
    Descriptor metadata are inserted as leading columns in the returned dataframe.

    Args:
        nuclei_labels (np.ndarray): 3D label image where each nucleus has a unique integer label.
        lif_image (np.ndarray): Multichannel image array indexed as ``lif_image[channel]``.
        markers (list[tuple[str, int, str]]): Marker definitions as
            ``(marker_name, channel_index, marker_role_or_descriptor)``.
        descriptor_dict (dict[str, str | int | float | bool]): Metadata values to prepend as columns.

    Returns:
        pd.DataFrame: Per-nucleus feature table containing morphology, per-marker
            intensity features, and descriptor metadata.
    """
    # Compute base morphology properties table for each nucleus
    props_morphology = regionprops_table(
        label_image=nuclei_labels,
        properties=MORPHOLOGY_PROPERTIES,
    )
    # Convert the properties dictionary to a DataFrame
    props_df = pd.DataFrame(props_morphology)

    # Iterate over all markers to extract intensity features
    for marker_name, ch_nr, *_ in markers:
        # Skip the brightfield marker (no intensity features required)
        if marker_name == "brightfield":
            continue

        # Compute intensity features for this marker channel
        props = regionprops_table(
            label_image=nuclei_labels,
            intensity_image=lif_image[ch_nr],  # The marker's image channel
            properties=INTENSITY_PROPERTIES,
        )
        intensity_df = pd.DataFrame(props)

        # Construct a renaming map for the intensity columns to include the marker's name
        rename_map = {"label": "label"}
        for prop in INTENSITY_PROPERTIES:
            if prop == "label":
                continue
            # For each intensity property, add the marker_name as a prefix
            if prop.startswith("intensity_"):
                suffix = prop.replace("intensity_", "")
                rename_map[prop] = f"{marker_name}_{suffix}_int"

        # Rename columns in the intensity DataFrame
        intensity_df.rename(columns=rename_map, inplace=True)
        # Merge the current marker's intensity features into the main DataFrame
        props_df = props_df.merge(intensity_df, on="label")

        # Derived columns section (per-marker columns like markerX_sum_int)
        mean_col = rename_map["intensity_mean"]
        area_col = "area"
        # Calculate total marker content per cell (mean_intensity * area)
        props_df[f"{marker_name}_sum_int"] = props_df[mean_col] * props_df[area_col]

    # Insert metadata columns at the beginning of the DataFrame, preserving input order
    insertion_position = 0
    for key, value in descriptor_dict.items():
        props_df.insert(insertion_position, key, value)
        insertion_position += 1

    # Return the final DataFrame with morphology, marker intensity features, and metadata
    return props_df