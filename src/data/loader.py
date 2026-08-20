"""
Data Loader for Multi-Label Datasets
Supports ARFF and MULAN XML formats.
"""

import os
import xml.etree.ElementTree as ET
import arff
import numpy as np
import pandas as pd


DATASET_CONFIG = {
    "emotions": {
        "arff_path": "data/emotions/emotions.arff",
        "xml_path": "data/emotions/emotions.xml",
        "label_location": "end",
        "num_labels": 6,
        "description": "Music emotion classification (6 labels, 593 samples)"
    },
    "music": {
        "arff_path": "data/Music.arff",
        "xml_path": None,
        "label_location": "start",
        "num_labels": 6,
        "description": "Music genre/emotion classification (6 labels, 592 samples)"
    },
    "scene": {
        "arff_path": "data/Scene.arff",
        "xml_path": None,
        "label_location": "start",
        "num_labels": 6,
        "description": "Image scene classification (6 labels, 2407 samples)"
    },
    "yeast": {
        "arff_path": "data/Yeast.arff",
        "xml_path": None,
        "label_location": "start",
        "num_labels": 14,
        "description": "Yeast gene functional classification (14 labels, 2417 samples)"
    },
    "cal500": {
        "arff_path": "data/CAL500.arff",
        "xml_path": None,
        "label_location": "end",
        "num_labels": 174,
        "description": "Music audio annotation (174 labels, 502 samples)"
    },
    "bibtex": {
        "arff_path": "data/bibtex/bibtex.arff",
        "xml_path": "data/bibtex/bibtex.xml",
        "label_location": "end",
        "num_labels": 159,
        "description": "BibTeX text tag recommendation (159 labels, 7395 samples)"
    },
    "enron": {
        "arff_path": "data/enron/enron.arff",
        "xml_path": "data/enron/enron.xml",
        "label_location": "end",
        "num_labels": 53,
        "description": "Email categorization (53 labels, 1702 samples)"
    },
    "genbase": {
        "arff_path": "data/genbase/genbase.arff",
        "xml_path": "data/genbase/genbase.xml",
        "label_location": "end",
        "num_labels": 27,
        "description": "Genomic sequence classification (27 labels, 662 samples)"
    },
    "medical": {
        "arff_path": "data/medical/medical.arff",
        "xml_path": "data/medical/medical.xml",
        "label_location": "end",
        "num_labels": 45,
        "description": "Medical clinical text categorization (45 labels, 978 samples)"
    },
    "reuters-k500": {
        "arff_path": "data/REUTERS-K500-EX2.arff",
        "xml_path": None,
        "label_location": "start",
        "num_labels": 103,
        "description": "Reuters text classification with 500 features (103 labels, 6000 samples)"
    }
}


def _parse_xml_labels(xml_path):
    """Extract list of label names from MULAN XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for elem in root.iter():
        if "name" in elem.attrib:
            labels.append(elem.attrib["name"])
    return labels


def _clean_binary_series(series):
    """Convert binary series containing {0, 1, '0', '1', 'YES', 'NO', 'TRUE', 'FALSE'} to int 0/1."""
    mapping = {
        1: 1, 0: 0,
        1.0: 1, 0.0: 0,
        "1": 1, "0": 0,
        "1.0": 1, "0.0": 0,
        "TRUE": 1, "FALSE": 0,
        "true": 1, "false": 0,
        "True": 1, "False": 0,
        "YES": 1, "NO": 0,
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "Y": 1, "N": 0
    }
    cleaned = series.map(mapping)
    # If unmapped values exist, convert numeric or default to 0
    if cleaned.isna().any():
        numeric_vals = pd.to_numeric(series, errors='coerce').fillna(0)
        cleaned = (numeric_vals > 0).astype(int)
    return cleaned.astype(np.int32).to_numpy()


def _process_features_dataframe(df_features):
    """
    Encode feature dataframe to float32 NumPy array.
    Handles numeric, binary strings ('YES'/'NO'), and categorical columns.
    """
    processed_cols = []
    
    for col in df_features.columns:
        s = df_features[col]
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(s):
            vals = s.fillna(0.0).to_numpy(dtype=np.float32)
            processed_cols.append(vals.reshape(-1, 1))
        else:
            # Check if binary string ('YES'/'NO', '0'/'1', 'true'/'false')
            unique_vals = set(s.dropna().astype(str).unique())
            if unique_vals.issubset({'0', '1', '0.0', '1.0', 'YES', 'NO', 'yes', 'no', 'TRUE', 'FALSE', 'true', 'false'}):
                vals = _clean_binary_series(s).astype(np.float32)
                processed_cols.append(vals.reshape(-1, 1))
            else:
                # Categorical column with multiple categories -> Factorize / Frequency / One-hot
                codes, _ = pd.factorize(s.fillna('MISSING'))
                vals = codes.astype(np.float32)
                processed_cols.append(vals.reshape(-1, 1))
                
    X = np.hstack(processed_cols).astype(np.float32)
    # Replace any NaN or Inf with 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_dataset(dataset_name, base_dir="."):
    """
    Load a multi-label dataset by name.

    Parameters:
        dataset_name (str): Key in DATASET_CONFIG.
        base_dir (str): Base workspace directory.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features), dtype float32
        Y (np.ndarray): Binary label matrix of shape (n_samples, n_labels), dtype int32
        feature_names (list): List of feature attribute names
        label_names (list): List of label attribute names
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset_name]
    arff_path = os.path.join(base_dir, config["arff_path"])
    xml_path = os.path.join(base_dir, config["xml_path"]) if config["xml_path"] else None

    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"ARFF file not found: {arff_path}")

    # Load ARFF file
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        arff_data = arff.load(f)

    attr_names = [a[0] for a in arff_data["attributes"]]
    df = pd.DataFrame(arff_data["data"], columns=attr_names)

    # Determine labels
    if xml_path and os.path.exists(xml_path):
        xml_labels = _parse_xml_labels(xml_path)
        # Verify if xml labels match columns in ARFF
        matched_labels = [l for l in xml_labels if l in df.columns]
        if len(matched_labels) == len(xml_labels):
            label_cols = matched_labels
        else:
            # Fallback to last num_labels
            num_labels = config["num_labels"]
            label_cols = attr_names[-num_labels:]
    else:
        num_labels = config["num_labels"]
        if config["label_location"] == "start":
            label_cols = attr_names[:num_labels]
        else:
            label_cols = attr_names[-num_labels:]

    feature_cols = [c for c in attr_names if c not in label_cols]

    # Process X (features)
    df_features = df[feature_cols]
    X = _process_features_dataframe(df_features)

    # Process Y (labels)
    df_labels = df[label_cols]
    Y_cols = []
    for col in label_cols:
        y_col = _clean_binary_series(df_labels[col])
        Y_cols.append(y_col.reshape(-1, 1))
    Y = np.hstack(Y_cols).astype(np.int32)

    return X, Y, feature_cols, label_cols


def load_all_datasets(base_dir="."):
    """Load all configured datasets into a dictionary."""
    data_dict = {}
    for name in DATASET_CONFIG:
        X, Y, f_names, l_names = load_dataset(name, base_dir=base_dir)
        data_dict[name] = {
            "X": X,
            "Y": Y,
            "feature_names": f_names,
            "label_names": l_names,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_labels": Y.shape[1],
            "cardinality": float(Y.sum(axis=1).mean()),
            "density": float(Y.mean())
        }
    return data_dict
