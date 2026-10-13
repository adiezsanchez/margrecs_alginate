import numpy as np

def extract_condition(container_id: str) -> str:
    """
    Extract the experimental condition from the provided container ID.

    Args:
        container_id (str): Identifier of the container, typically including condition label.

    Returns:
        str: Experimental condition. Possible values:
            - "perfusion" if 'perfusion' is in the ID (case-insensitive)
            - "static" if 'static' is in the ID (case-insensitive)
            - "undefined" if neither condition is found in the ID

    Notes:
        This function is case-insensitive and will return "undefined" if no known condition tag is present.
    """
    cid = str(container_id).lower()
    if "perfusion" in cid:
        return "perfusion"
    elif "static" in cid:
        return "static"
    else:
        return "undefined"

def extract_condition_coc(image_name: str) -> str:
    """
    Extract the experimental condition from the COC image name.

    Args:
        image_name (str): Image/CSV stem, expected to contain ``+FGF`` or ``-FGF``
            (e.g. ``D1SD1+FGF-1.1-10x``, ``D1SD1-FGF-1.1-10x``).

    Returns:
        str: Experimental condition. Possible values:
            - "+FGF" if '+FGF' is in the name (case-insensitive)
            - "-FGF" if '-FGF' is in the name (case-insensitive)
            - "undefined" if neither condition is found

    Notes:
        Matching is case-insensitive. ``+FGF`` is checked before ``-FGF`` so names
        like ``D3-SD2-+FGF-...`` resolve to ``+FGF``.
    """
    name = str(image_name).lower()
    if "+fgf" in name:
        return "+FGF"
    elif "-fgf" in name:
        return "-FGF"
    else:
        return "undefined"

def extract_day(image_name: str) -> int:
    """
    Extract the day number from the image file name.

    Args:
        image_name (str): Name or path of the image file, expected to contain tags such as "d3-", "d7-", or "d14-".

    Returns:
        int: Day number (3, 7, or 14) if found, else 0 if the day tag is not found.
    """
    name = str(image_name)
    if any(tag in name for tag in ["d3-", "D3-"]):
        return 3
    elif any(tag in name for tag in ["d7-", "D7-"]):
        return 7
    elif any(tag in name for tag in ["d14-", "D14-"]):
        return 14
    else:
        return 0

def extract_day_coc(image_name: str) -> int:
    """
    Extract the day number from the image file name for COC data.

    Args:
        image_name (str): Name or path of the image file, expected to contain tags such as "day1" or "day3".

    Returns:
        int: Day number (1 or 3) if found, else 0 if the day tag is not found.
    """
    name = str(image_name)
    if any(tag in name for tag in ["day1", "Day1"]):
        return 1
    elif any(tag in name for tag in ["day3", "Day3"]):
        return 3
    else:
        return 0


def extract_hours_prior_coc(container_id: str) -> int:
    """
    Extract the hours prior to HPFa from the COC container ID.

    Args:
        container_id (str): Container folder id, expected to contain a tag like
            ``24hpriortoHPFa`` or ``48hpriortoHPFa``
            (e.g. ``Co-culture-day1-A549-24hpriortoHPFa``).

    Returns:
        int: Hours prior to HPFa (24 or 48) if found, else 0 if not found.
    """
    cid = str(container_id).lower()
    tag = "hpriortohpfa"
    if tag in cid:
        return int(cid.split(tag)[0].split("-")[-1])
    else:
        return 0

def extract_magnification(image_name: str) -> int:
    """
    Extract the magnification level from the image file name.

    Args:
        image_name (str): Name or path of the image file, expected to contain tags such as "-5x-", "-10x-", or "-20x-".

    Returns:
        int: Magnification (5, 10, or 20) if found, else 0 if not found.
    """
    name = str(image_name).lower()
    if "-5x-" in name:
        return 5
    elif "-10x-" in name:
        return 10
    elif "-20x-" in name:
        return 20
    else:
        return 0


def classify_ki67_status_kmeans(
    props_df: "pd.DataFrame",
    feature_columns: list = ['ki67_mean_int', 'ki67_sum_int']
) -> "pd.DataFrame":
    """
    Classify nuclei as Ki67 positive (1) or negative (0) using k-means clustering
    on the specified ki67 intensity features.

    Args:
        props_df (pd.DataFrame): DataFrame containing per-nucleus features.
        feature_columns (list, optional): List of feature column names to use for clustering.
            Default is ['ki67_mean_int', 'ki67_sum_int'].

    Returns:
        pd.DataFrame: The input DataFrame with an additional column:
            - 'ki67_status': 1 for Ki67 positive, 0 for Ki67 negative

    Notes:
        - Features are standardized before clustering.
        - The cluster with HIGHER mean 'ki67_mean_int' is mapped to Ki67 positive (1).
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import numpy as np

    # Select features for clustering
    features = props_df[feature_columns].values

    # Standardize features to have mean=0, std=1
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform k-means clustering (k=2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features_scaled)

    # Add cluster assignments to DataFrame
    props_df = props_df.copy()
    props_df['ki67_cluster_id'] = cluster_ids

    # Determine which cluster has higher ki67_mean_int
    cluster_means = (
        props_df.groupby('ki67_cluster_id')['ki67_mean_int']
        .mean()
        .sort_values()
    )
    ordered_cluster_ids = cluster_means.index.tolist()
    ki67_status_map = {
        ordered_cluster_ids[0]: 0,  # lower mean: negative
        ordered_cluster_ids[1]: 1   # higher mean: positive
    }
    props_df['ki67_status'] = props_df['ki67_cluster_id'].map(ki67_status_map)
    props_df.drop(columns=['ki67_cluster_id'], inplace=True)

    return props_df