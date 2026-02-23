"""Utility helpers for the Diabetes Selection project.

Includes data preprocessing, TF feature-column construction, and
Bayesian neural-network layer factories (adapted from TensorFlow
Probability tutorials).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from student_utils import create_tf_numeric_feature


# ---------------------------------------------------------------------------
# Dataset aggregation
# ---------------------------------------------------------------------------
def aggregate_dataset(
    df: pd.DataFrame,
    grouping_field_list: List[str],
    array_field: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Group *df* and one-hot encode *array_field* values.

    Args:
        df: Input DataFrame.
        grouping_field_list: Columns to group by.
        array_field: Column whose values are collected into lists and then
            one-hot encoded.

    Returns:
        Tuple of (concatenated DataFrame, list of dummy column names).
    """
    df = (
        df.groupby(grouping_field_list)["encounter_id", array_field]
        .apply(lambda x: x[array_field].values.tolist())
        .reset_index()
        .rename(columns={0: array_field + "_array"})
    )

    dummy_df = pd.get_dummies(
        df[array_field + "_array"].apply(pd.Series).stack()
    ).sum(level=0)
    dummy_col_list = [x.replace(" ", "_") for x in list(dummy_df.columns)]
    mapping_name_dict = dict(
        zip(list(dummy_df.columns), dummy_col_list)
    )
    concat_df = pd.concat([df, dummy_df], axis=1)
    concat_df.columns = [x.replace(" ", "_") for x in list(concat_df.columns)]

    return concat_df, dummy_col_list


# ---------------------------------------------------------------------------
# Column casting & imputation
# ---------------------------------------------------------------------------
def cast_df(df: pd.DataFrame, col: str, d_type: type = str) -> pd.Series:
    """Cast a single column to *d_type*."""
    return df[col].astype(d_type)


def impute_df(
    df: pd.DataFrame, col: str, impute_value: float = 0
) -> pd.Series:
    """Fill missing values in *col* with *impute_value*."""
    return df[col].fillna(impute_value)


def preprocess_df(
    df: pd.DataFrame,
    categorical_col_list: List[str],
    numerical_col_list: List[str],
    predictor: str,
    categorical_impute_value: str = "nan",
    numerical_impute_value: float = 0,
) -> pd.DataFrame:
    """Cast categorical columns to str, impute numerics, and ensure the
    predictor is float.

    Args:
        df: Input DataFrame (modified in-place).
        categorical_col_list: Columns to cast as strings.
        numerical_col_list: Columns to impute with *numerical_impute_value*.
        predictor: Target column, cast to float.
        categorical_impute_value: (unused, kept for API compat).
        numerical_impute_value: Fill value for missing numerics.

    Returns:
        The preprocessed DataFrame.
    """
    df[predictor] = df[predictor].astype(float)
    for c in categorical_col_list:
        df[c] = cast_df(df, c, d_type=str)
    for numerical_column in numerical_col_list:
        df[numerical_column] = impute_df(df, numerical_column, numerical_impute_value)
    return df


# ---------------------------------------------------------------------------
# TF Dataset construction
# ---------------------------------------------------------------------------
def df_to_dataset(
    df: pd.DataFrame, predictor: str, batch_size: int = 32
) -> tf.data.Dataset:
    """Convert a pandas DataFrame into a shuffled, batched ``tf.data.Dataset``.

    Adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
    """
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


# ---------------------------------------------------------------------------
# Vocabulary file helpers
# ---------------------------------------------------------------------------
def write_vocabulary_file(
    vocab_list: np.ndarray,
    field_name: str,
    default_value: str,
    vocab_dir: str = "./diabetes_vocab/",
) -> str:
    """Write a vocabulary text file for a categorical feature column.

    Creates *vocab_dir* if it does not already exist.

    Args:
        vocab_list: Unique category values.
        field_name: Column / feature name (used in the filename).
        default_value: Inserted as the first row (TF requirement).
        vocab_dir: Target directory for vocabulary files.

    Returns:
        Path to the written vocabulary file.
    """
    os.makedirs(vocab_dir, exist_ok=True)
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0)
    pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path


def build_vocab_files(
    df: pd.DataFrame,
    categorical_column_list: List[str],
    default_value: str = "00",
) -> List[str]:
    """Build vocabulary files for every categorical column.

    Args:
        df: Training DataFrame.
        categorical_column_list: Columns to create vocab files for.
        default_value: Default / OOV placeholder value.

    Returns:
        List of written file paths.
    """
    vocab_files_list: List[str] = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list


# ---------------------------------------------------------------------------
# Quick EDA
# ---------------------------------------------------------------------------
def show_group_stats_viz(df: pd.DataFrame, group: str) -> None:
    """Print and plot group sizes for a given column."""
    print(df.groupby(group).size())
    print(df.groupby(group).size().plot(kind="barh"))


# ---------------------------------------------------------------------------
# TF Probability layer factories
# ---------------------------------------------------------------------------
# Adapted from:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/
# examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb


def posterior_mean_field(
    kernel_size: int, bias_size: int = 0, dtype: tf.DType = None
) -> tf.keras.Sequential:
    """Return a mean-field variational posterior for a Bayesian layer."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:]),
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_trainable(
    kernel_size: int, bias_size: int = 0, dtype: tf.DType = None
) -> tf.keras.Sequential:
    """Return a trainable prior for a Bayesian layer."""
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Feature demo / stats
# ---------------------------------------------------------------------------
def demo(
    feature_column: tf.feature_column.FeatureColumn,
    example_batch: Dict[str, tf.Tensor],
) -> tf.Tensor:
    """Print and return the output of a single feature column on a batch."""
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)


def calculate_stats_from_train_data(
    df: pd.DataFrame, col: str
) -> Tuple[float, float]:
    """Return (mean, std) for *col*, calling ``.describe()`` only once."""
    stats = df[col].describe()
    return stats["mean"], stats["std"]


def create_tf_numerical_feature_cols(
    numerical_col_list: List[str], train_df: pd.DataFrame
) -> List[tf.feature_column.NumericColumn]:
    """Build normalised numeric TF feature columns from training data stats.

    Args:
        numerical_col_list: Numeric column names.
        train_df: Training DataFrame used to compute mean/std.

    Returns:
        List of ``tf.feature_column.NumericColumn`` objects.
    """
    tf_numeric_col_list: List[tf.feature_column.NumericColumn] = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list
