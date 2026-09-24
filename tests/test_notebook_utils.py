import importlib.util
import warnings
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.svm import SVC


_UTILS_PATH = Path(__file__).parents[1] / "notebooks" / "_hypothesis_utils.py"
_SPEC = importlib.util.spec_from_file_location("hypothesis_utils", _UTILS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["hypothesis_utils"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_adaptive_embedding_dimension_uses_twenty_times_input_width_by_default():
    assert _MODULE.adaptive_embedding_dimension(7) == 140
    assert _MODULE.adaptive_embedding_dimension(np.int64(7), dimension_ratio=3) == 21


def test_openml_numeric_split_imputes_missing_values_from_training_rows():
    dataset = _MODULE.OpenMLDataset(
        task_id=1,
        dataset_id=1,
        name="numeric-with-missing",
        X=np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
        y=np.array([0, 0, 1, 1]),
        feature_names=("x1", "x2"),
        target_name="target",
        n_original=4,
        n_used=4,
    )

    X_train, X_test, _, _, preprocessor = _MODULE.openml_classification_split(
        dataset, test_size=0.5, seed=0
    )

    assert preprocessor is not None
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()


def test_sklearn_high_cardinality_classification_warning_is_scoped_out():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.arange(40)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _MODULE.suppress_sklearn_classification_target_warning():
            SVC().fit(X, y)

    assert not any(
        "The number of unique classes is greater than 50%" in str(warning.message)
        for warning in caught
    )


@pytest.mark.parametrize("n_features", [0, -1, 1.5])
def test_adaptive_embedding_dimension_rejects_invalid_width(n_features):
    with pytest.raises(ValueError, match="n_features"):
        _MODULE.adaptive_embedding_dimension(n_features)
