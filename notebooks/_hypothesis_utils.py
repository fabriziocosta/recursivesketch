"""Small, reproducible helpers shared by the hypothesis notebooks."""

from __future__ import annotations

import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted

# Keep notebook imports working when Jupyter starts in ``notebooks/`` rather
# than from an installed editable package.
repository_root = Path(__file__).resolve().parent.parent
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from recursivesketch import DecisionPathEncoder, RecursiveSketchClassifier, make_projector


CLASSIFICATION_SEEDS = (0, 1, 2)
DEFAULT_N_ESTIMATORS = 100
DEFAULT_DIMENSION_RATIO = 20


def adaptive_embedding_dimension(
    n_features,
    dimension_ratio=DEFAULT_DIMENSION_RATIO,
):
    """Return ``ceil(dimension_ratio * p)`` for a positive ratio."""
    if not isinstance(n_features, (int, np.integer)) or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    if (
        isinstance(dimension_ratio, (bool, np.bool_))
        or not isinstance(dimension_ratio, Real)
        or not np.isfinite(dimension_ratio)
        or dimension_ratio <= 0
    ):
        raise ValueError("dimension_ratio must be a positive finite number")
    return max(1, int(np.ceil(dimension_ratio * n_features)))


@dataclass
class OpenMLDataset:
    """A reproducibly selected OpenML task and its sampled observations."""

    task_id: int
    dataset_id: int
    name: str
    X: object
    y: np.ndarray
    feature_names: tuple
    target_name: Optional[str]
    n_original: int
    n_used: int


def make_datasets(
    n=4,
    max_size=500,
    seed=42,
    suite_name="OpenML-CC18",
    task_ids=None,
    dataset_names=None,
):
    """Load a deterministic subset of OpenML-CC18 classification datasets.

    Parameters
    ----------
    n:
        Maximum number of tasks to load. Tasks are selected by ascending OpenML
        task ID unless ``task_ids`` is supplied.
    max_size:
        Maximum number of observations retained per dataset. Larger datasets
        are sampled with stratification on the target. ``None`` keeps all rows.
    seed:
        Seed used for stratified subsampling. The same seed produces the same
        observations for every selected task.
    suite_name:
        OpenML benchmark-suite alias. The default is OpenML-CC18.
    task_ids:
        Optional explicit OpenML task IDs. This is useful for a fixed paper
        benchmark; ``n`` still limits the number of IDs used.
    dataset_names:
        Optional exact OpenML dataset names. Names are resolved within the
        selected suite and returned in the order supplied. When provided,
        ``n`` is ignored so that every requested name is loaded. Cannot be
        combined with ``task_ids``.

    Returns
    -------
    list[OpenMLDataset]
        Dataset objects containing stable identifiers, metadata, and sampled
        feature/target arrays.
    """
    if n is not None and (not isinstance(n, (int, np.integer)) or n <= 0):
        raise ValueError("n must be a positive integer or None")
    if max_size is not None and (
        not isinstance(max_size, (int, np.integer)) or max_size <= 0
    ):
        raise ValueError("max_size must be a positive integer or None")

    try:
        import openml
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "OpenML support requires the notebook dependencies; install "
            "with `python -m pip install -e '.[notebook]'`."
        ) from exc

    if task_ids is not None and dataset_names is not None:
        raise ValueError("task_ids and dataset_names cannot be combined")

    suite = openml.study.get_suite(suite_name)
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        requested_names = list(dict.fromkeys(str(name) for name in dataset_names))
        if not requested_names:
            raise ValueError("dataset_names must contain at least one name")

        task_ids_by_name = {}
        suite_task_ids = {int(task_id) for task_id in suite.tasks}
        for name in requested_names:
            task_listing = openml.tasks.list_tasks(
                data_name=name,
                output_format="dataframe",
            )
            matching_tasks = task_listing[
                task_listing["tid"].astype(int).isin(suite_task_ids)
            ]
            if not matching_tasks.empty:
                task_ids_by_name[name] = int(matching_tasks.iloc[0]["tid"])
        missing_names = [name for name in requested_names if name not in task_ids_by_name]
        if missing_names:
            raise ValueError(
                "Dataset names not found in "
                f"{suite_name}: {', '.join(missing_names)}"
            )
        selected_task_ids = [task_ids_by_name[name] for name in requested_names]
    else:
        selected_task_ids = sorted(int(task_id) for task_id in (task_ids or suite.tasks))
        if n is not None:
            selected_task_ids = selected_task_ids[:n]
    if not selected_task_ids:
        raise ValueError(f"OpenML suite {suite_name!r} contains no tasks")

    datasets = []
    for task_id in selected_task_ids:
        task = openml.tasks.get_task(task_id, download_data=True)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        y = np.asarray(y).reshape(-1)
        n_original = len(y)
        if max_size is not None and n_original > max_size:
            from sklearn.model_selection import train_test_split

            selected, _ = train_test_split(
                np.arange(n_original),
                train_size=max_size,
                stratify=y,
                random_state=seed,
            )
            selected = np.sort(selected)
            X = X.iloc[selected] if hasattr(X, "iloc") else X[selected]
            y = y[selected]

        dataset = task.get_dataset()
        feature_names = tuple(
            getattr(X, "columns", getattr(dataset, "features", []))
        )
        target_name = getattr(task, "target_name", None)
        if target_name is None:
            target_name = getattr(dataset, "default_target_attribute", None)
        datasets.append(
            OpenMLDataset(
                task_id=int(task_id),
                dataset_id=int(task.dataset_id),
                name=str(getattr(dataset, "name", task_id)),
                X=X,
                y=y,
                feature_names=feature_names,
                target_name=target_name,
                n_original=n_original,
                n_used=len(y),
            )
        )
    return datasets


def openml_classification_split(dataset, test_size=0.3, seed=0):
    """Encode one ``OpenMLDataset`` after making a stratified holdout split.

    The preprocessing is fitted on the training rows only. Numeric columns
    receive median imputation and categorical columns receive most-frequent
    imputation followed by one-hot encoding. Numpy inputs are validated and
    passed through as floating-point arrays.
    """
    from sklearn.model_selection import train_test_split

    X, y = dataset.X, np.asarray(dataset.y).reshape(-1)
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    if hasattr(X, "iloc"):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        X_train_raw = X.iloc[train_indices]
        X_test_raw = X.iloc[test_indices]
        categorical = list(X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns)
        numeric = [column for column in X_train_raw.columns if column not in categorical]
        transformers = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    SimpleImputer(strategy="median"),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                    categorical,
                )
            )
        preprocessor = ColumnTransformer(transformers, sparse_threshold=0.3)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
    else:
        from sklearn.impute import SimpleImputer

        preprocessor = SimpleImputer(strategy="median")
        X_train = preprocessor.fit_transform(X[train_indices])
        X_test = preprocessor.transform(X[test_indices])
        X_train = check_array(X_train, dtype=np.float64, ensure_2d=True)
        X_test = check_array(X_test, dtype=np.float64, ensure_2d=True)

    return (
        X_train,
        X_test,
        y[train_indices],
        y[test_indices],
        preprocessor,
    )


def classification_data(seed=42, n_samples=2200, n_features=120):
    """Return a fixed train/validation/test split for a difficult classification task."""
    n_informative = max(2, min(n_features - 1, 12))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=min(24, n_features - n_informative),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=0.65,
        flip_y=0.04,
        weights=[0.55, 0.45],
        random_state=seed,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=seed + 1000,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_data(seed=42, n_samples=1800, n_features=80):
    """Return a fixed train/validation/test split for a noisy regression task."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=16,
        n_targets=1,
        noise=18.0,
        random_state=seed,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=seed + 1000
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def forest_classifier(seed, n_estimators=DEFAULT_N_ESTIMATORS):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        n_jobs=-1,
        random_state=seed,
    )


def forest_regressor(seed, n_estimators=DEFAULT_N_ESTIMATORS):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        n_jobs=-1,
        random_state=seed,
    )


def downstream_classifier(seed=42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )


def openml_downstream_classifier(seed=42):
    """Sparse-compatible downstream classifier for encoded OpenML features."""
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=3000, random_state=seed),
    )


def openml_svm_rbf_classifier(seed=42):
    """Sparse-compatible RBF SVM with automatic feature scaling.

    ``C=1`` is the fixed default SVC penalty, corresponding to the common
    auto-regularization choice lambda=1/n_samples under the averaged-loss
    convention. ``gamma='scale'`` derives the RBF width from the input data.
    """
    return make_pipeline(
        StandardScaler(with_mean=False),
        SVC(C=1.0, gamma="scale", kernel="rbf", random_state=seed),
    )


def openml_output_classifier(
    classifier="random_forest",
    seed=42,
    n_estimators=DEFAULT_N_ESTIMATORS,
):
    """Construct a downstream classifier for an OpenML Recursive Sketch output."""
    if classifier == "logistic_regression":
        return openml_downstream_classifier(seed)
    if classifier == "random_forest":
        return forest_classifier(seed, n_estimators=n_estimators)
    if classifier == "svm_rbf_auto_lambda":
        return openml_svm_rbf_classifier(seed)
    raise ValueError(
        "classifier must be one of 'logistic_regression', 'random_forest', "
        "or 'svm_rbf_auto_lambda'"
    )


def recursive_partition_classifier(seed, n_estimators=DEFAULT_N_ESTIMATORS):
    """Construct the bootstrapped recursive-partition path estimator."""
    try:
        from recursive_partition import (
            BaggedRecursivePartitionClassifier,
            RecursivePartitionClassifier,
        )
    except ModuleNotFoundError:
        search_roots = (Path.cwd(), *Path.cwd().parents)
        sister_candidates = [
            root / "RecursiveParitionClassifier" for root in search_roots
        ]
        sister_repository = next(
            (
                candidate
                for candidate in sister_candidates
                if (candidate / "recursive_partition").exists()
            ),
            None,
        )
        if sister_repository is None:
            raise ModuleNotFoundError(
                "Install the RecursiveParitionClassifier sister repository or "
                "place it next to RecursiveSketch."
            )
        sys.path.insert(0, str(sister_repository))
        from recursive_partition import (
            BaggedRecursivePartitionClassifier,
            RecursivePartitionClassifier,
        )

    return BaggedRecursivePartitionClassifier(
        estimator=RecursivePartitionClassifier(),
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=seed,
    )


def downstream_regressor():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def initial_projection(X_train, X_test, n_components, seed, projection_type="sparse"):
    projector = make_projector(n_components, projection_type, seed)
    projector.fit(X_train)
    return projector.transform(X_train), projector.transform(X_test), projector


class OneShotTreePathTransformer(BaseEstimator, TransformerMixin):
    """One forest plus one path projection, used as the non-iterative baseline."""

    def __init__(self, estimator, n_components=32, random_state=None):
        self.estimator = estimator
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.forest_ = clone(self.estimator).fit(X, y)
        self.encoder_ = DecisionPathEncoder().fit(self.forest_)
        V = self.encoder_.transform(X)
        self.projector_ = make_projector(
            self.n_components, "gaussian", self.random_state
        ).fit(V)
        return self

    def transform(self, X):
        check_is_fitted(self, ("forest_", "encoder_", "projector_"))
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        return self.projector_.transform(self.encoder_.transform(X))


class NoConcatRecursiveSketch(BaseEstimator, TransformerMixin):
    """Iteration ablation that replaces X with Z instead of concatenating."""

    def __init__(self, estimator, n_components=32, n_iterations=1, random_state=None):
        self.estimator = estimator
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        seeds = np.random.RandomState(self.random_state).randint(
            0, np.iinfo(np.int32).max, size=self.n_iterations + 1
        )
        self.initial_projector_ = make_projector(
            self.n_components, "sparse", seeds[0]
        ).fit(X)
        current = self.initial_projector_.transform(X)
        self.encoders_ = []
        self.projectors_ = []
        for iteration in range(self.n_iterations):
            forest = clone(self.estimator).fit(current, y)
            encoder = DecisionPathEncoder().fit(forest)
            V = encoder.transform(current)
            projector = make_projector(
                self.n_components, "gaussian", seeds[iteration + 1]
            ).fit(V)
            current = projector.transform(V)
            self.encoders_.append(encoder)
            self.projectors_.append(projector)
        return self

    def transform(self, X):
        check_is_fitted(self, ("initial_projector_", "encoders_"))
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        current = self.initial_projector_.transform(X)
        for encoder, projector in zip(self.encoders_, self.projectors_):
            current = projector.transform(encoder.transform(current))
        return current


class MatchedRandomSparsePathTransformer(BaseEstimator, TransformerMixin):
    """Control with random sparse features matched to the real path matrix shape."""

    def __init__(self, estimator, n_components=32, random_state=None):
        self.estimator = estimator
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.forest_ = clone(self.estimator).fit(X, y)
        self.encoder_ = DecisionPathEncoder().fit(self.forest_)
        V = self.encoder_.transform(X)
        self.n_path_features_ = V.shape[1]
        self.nnz_per_row_ = max(1, int(round(V.nnz / V.shape[0])))
        random_V = self._random_sparse_features(V.shape[0])
        self.projector_ = make_projector(
            self.n_components, "gaussian", self.random_state
        ).fit(random_V)
        return self

    def _random_sparse_features(self, n_rows):
        rng = np.random.RandomState(self.random_state)
        rows = np.repeat(np.arange(n_rows), self.nnz_per_row_)
        cols = rng.randint(0, self.n_path_features_, size=rows.size)
        signs = rng.choice(np.array([-1.0, 1.0]), size=rows.size)
        return sparse.coo_matrix(
            (signs, (rows, cols)),
            shape=(n_rows, self.n_path_features_),
        ).tocsr()

    def transform(self, X):
        check_is_fitted(self, ("forest_", "encoder_", "projector_"))
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        V = self.encoder_.transform(X)
        random_V = self._random_sparse_features(V.shape[0])
        return self.projector_.transform(random_V)


def full_sketch(
    seed,
    n_components=32,
    n_iterations=2,
    kind="classifier",
    n_estimators=DEFAULT_N_ESTIMATORS,
    dimension_ratio=None,
    dimension_mode="expanding",
    output_format="auto",
    initial_projection_type="sparse",
    path_projection_type="gaussian",
    concat_projection_type="sparse",
    normalization="none",
    path_estimator="random_forest",
):
    if path_estimator == "random_forest":
        estimator = (
            forest_classifier(seed, n_estimators=n_estimators)
            if kind == "classifier"
            else forest_regressor(seed, n_estimators=n_estimators)
        )
    elif path_estimator == "recursive_partition" and kind == "classifier":
        estimator = recursive_partition_classifier(seed, n_estimators=n_estimators)
    else:
        raise ValueError(
            "path_estimator must be 'random_forest', or "
            "'recursive_partition' for classification"
        )
    return RecursiveSketchClassifier(
        estimator=estimator,
        n_components=n_components,
        n_iterations=n_iterations,
        dimension_ratio=dimension_ratio,
        dimension_mode=dimension_mode,
        output_format=output_format,
        initial_projection_type=initial_projection_type,
        path_projection_type=path_projection_type,
        concat_projection_type=concat_projection_type,
        normalization=normalization,
        random_state=seed,
    )


recursive_sketch = full_sketch


@contextmanager
def suppress_sklearn_classification_target_warning():
    """Suppress sklearn's high-cardinality classification-target heuristic.

    OpenML tasks and recursive partition nodes can legitimately have many
    classes relative to the local sample count.  The warning is useful for
    catching accidental use of a classifier on continuous targets, but these
    helpers operate on explicitly selected classification tasks.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"The number of unique classes is greater than 50% of the "
                r"number of samples\..*"
            ),
            category=UserWarning,
            module=r"sklearn\.svm\._base",
        )
        yield


def timed_fit_transform(transformer, X, y):
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    transformed = transformer.fit_transform(X, y)
    return transformed, time.perf_counter() - start_wall, time.process_time() - start_cpu


def serialized_size_mb(value):
    return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)) / 2**20


def classification_score(model, X_train, y_train, X_test, y_test):
    with suppress_sklearn_classification_target_warning():
        model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return float(accuracy_score(y_test, prediction)), int(np.count_nonzero(prediction != y_test))


def regression_scores(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return float(r2_score(y_test, prediction)), float(mean_squared_error(y_test, prediction) ** 0.5)


def print_verdict(name, supported, evidence):
    label = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"{name}: {label} — {evidence}")


def critical_difference_stats(
    results,
    block_column,
    method_column,
    score_column="accuracy",
    higher_is_better=True,
):
    """Return paired scores, average ranks, and Nemenyi p-values for a CD plot.

    ``block_column`` may be a string or a list of columns. Passing
    ``["dataset_id", "repetition_id"]`` treats every dataset/repetition pair
    as one paired block, while retaining the original single-column API.
    """
    import scikit_posthocs as sp

    block_index = block_column
    wide = results.pivot(index=block_index, columns=method_column, values=score_column)
    if wide.isna().any().any():
        raise ValueError("critical-difference input must contain every method for every block")
    average_ranks = wide.rank(
        axis=1,
        ascending=not higher_is_better,
        method="average",
    ).mean()
    significance = sp.posthoc_nemenyi_friedman(wide)
    return wide, average_ranks, significance


def plot_critical_difference(
    results,
    block_column,
    method_column,
    score_column="accuracy",
    higher_is_better=True,
    ax=None,
    title=None,
):
    """Plot a critical-difference diagram from paired experiment results."""
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    _, average_ranks, significance = critical_difference_stats(
        results,
        block_column=block_column,
        method_column=method_column,
        score_column=score_column,
        higher_is_better=higher_is_better,
    )
    sp.critical_difference_diagram(
        average_ranks.to_dict(),
        significance,
        alpha=0.05,
        ax=ax,
        left_only=True,
    )
    if title is not None:
        ax.set_title(title)
    return ax
