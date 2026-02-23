import functools
from typing import List, Tuple

import numpy as np
import os
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
READMISSION_THRESHOLD = 5
TRAIN_RATIO = 0.6
TRAIN_VAL_RATIO = 0.8


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------
def reduce_dimension_ndc(df: pd.DataFrame, ndc_df: pd.DataFrame) -> pd.DataFrame:
    """Map NDC codes to proprietary drug names to reduce dimensionality.

    Note: the resulting column is called ``generic_drug_name`` for
    compatibility with downstream notebooks, but it actually contains the
    *proprietary* (brand) name from the NDC lookup table.

    Args:
        df: Input encounter dataset containing an ``ndc_code`` column.
        ndc_df: NDC drug-code reference table.

    Returns:
        DataFrame with an added ``generic_drug_name`` column (NDC columns
        dropped).
    """
    df1 = pd.merge(
        df,
        ndc_df[["Proprietary Name", "NDC_Code"]],
        left_on="ndc_code",
        right_on="NDC_Code",
    )
    df1["generic_drug_name"] = df1["Proprietary Name"]
    df1 = df1.drop(["NDC_Code", "Proprietary Name"], axis=1)
    return df1


# ---------------------------------------------------------------------------
# Encounter de-duplication
# ---------------------------------------------------------------------------
def select_first_encounter(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the first encounter per patient.

    The dataframe is sorted by ``encounter_id`` so that the earliest
    encounter appears first, then duplicates on ``patient_nbr`` are dropped.

    Args:
        df: DataFrame containing ``encounter_id`` and ``patient_nbr``.

    Returns:
        De-duplicated DataFrame with one row per patient.
    """
    first_encounter_df = df.copy()
    first_encounter_df = first_encounter_df.sort_values("encounter_id")
    first_encounter_df = first_encounter_df.drop_duplicates(
        subset=["patient_nbr"], keep="first"
    )
    return first_encounter_df


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
def patient_dataset_splitter(
    df: pd.DataFrame, patient_key: str = "patient_nbr"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / validation / test sets by patient.

    A fixed random seed (42) is used so that splits are reproducible.

    Args:
        df: Input dataset.
        patient_key: Column that uniquely identifies a patient.

    Returns:
        Tuple of (train, validation, test) DataFrames.
    """
    rng = np.random.RandomState(seed=42)
    df = pd.DataFrame(df)
    df = df.iloc[rng.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)

    sample_size_train = round(total_values * TRAIN_RATIO)
    sample_size_train_val = round(total_values * TRAIN_VAL_RATIO)

    train = df[df[patient_key].isin(unique_values[:sample_size_train])].reset_index(
        drop=True
    )
    validation = df[
        df[patient_key].isin(unique_values[sample_size_train:sample_size_train_val])
    ].reset_index(drop=True)
    test = df[
        df[patient_key].isin(unique_values[sample_size_train_val:])
    ].reset_index(drop=True)
    return train, validation, test


# ---------------------------------------------------------------------------
# TensorFlow categorical feature columns
# ---------------------------------------------------------------------------
def create_tf_categorical_feature_cols(
    categorical_col_list: List[str],
    vocab_dir: str = "./diabetes_vocab/",
) -> List[tf.feature_column.IndicatorColumn]:
    """Build TF indicator feature columns from vocabulary files.

    Args:
        categorical_col_list: Categorical column names.
        vocab_dir: Directory that contains ``<col>_vocab.txt`` files.

    Returns:
        List of TF ``IndicatorColumn`` objects.
    """
    output_tf_list: List[tf.feature_column.IndicatorColumn] = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir, c + "_vocab.txt")
        tf_categorical_feature_column = (
            tf.feature_column.categorical_column_with_vocabulary_file(
                key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1
            )
        )
        tf_categorical_feature_column = tf.feature_column.indicator_column(
            tf_categorical_feature_column
        )
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


# ---------------------------------------------------------------------------
# Numeric normalisation & feature columns
# ---------------------------------------------------------------------------
def normalize_numeric_with_zscore(
    col: tf.Tensor, mean: float, std: float
) -> tf.Tensor:
    """Z-score normalisation safe against zero standard deviation."""
    if std == 0:
        return col - mean
    return (col - mean) / std


def create_tf_numeric_feature(
    col: str, MEAN: float, STD: float, default_value: float = 0
) -> tf.feature_column.NumericColumn:
    """Create a normalised TF numeric feature column.

    Args:
        col: Column name.
        MEAN: Training-set mean for the column.
        STD: Training-set standard deviation for the column.
        default_value: Value used when the field is missing.

    Returns:
        A ``tf.feature_column.NumericColumn``.
    """
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col,
        default_value=default_value,
        normalizer_fn=normalizer,
        dtype=tf.float64,
    )
    return tf_numeric_feature


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def get_mean_std_from_preds(
    diabetes_yhat,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mean and standard deviation from a TF Probability prediction.

    Args:
        diabetes_yhat: TF Probability distribution prediction object.

    Returns:
        Tuple of (mean, stddev) arrays.
    """
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s


def get_student_binary_prediction(
    df: pd.DataFrame, pred_mean: str
) -> pd.Series:
    """Convert probability predictions to binary labels.

    A prediction is positive (1) when the mean is at or above
    ``READMISSION_THRESHOLD`` (default 5), and negative (0) otherwise.

    Args:
        df: DataFrame containing the prediction column.
        pred_mean: Name of the column holding the predicted mean.

    Returns:
        A Series of binary labels (0 or 1).
    """
    df["score"] = df[pred_mean].apply(
        lambda x: 1 if x >= READMISSION_THRESHOLD else 0
    )
    student_binary_prediction = df["score"]
    return student_binary_prediction
