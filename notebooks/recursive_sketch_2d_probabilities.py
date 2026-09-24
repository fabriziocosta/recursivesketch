"""Reusable experiment code for the Recursive Sketch 2-D probability demo."""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SISTER_ROOT = _REPOSITORY_ROOT.parent / "RecursiveParitionClassifier"
if not (_SISTER_ROOT / "recursive_partition").is_dir():
    raise ModuleNotFoundError(
        "Install RecursiveParitionClassifier or place it next to RecursiveSketch."
    )
if str(_SISTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SISTER_ROOT))

from recursive_partition import (  # noqa: E402
    BaggedRecursivePartitionClassifier,
    RecursivePartitionClassifier,
)
from recursivesketch import RecursiveSketchClassifier  # noqa: E402


@contextmanager
def suppress_sklearn_classification_target_warning():
    """Suppress sklearn's high-cardinality classification-target heuristic."""
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


DATASETS = (
    "circles",
    "moon",
    "spirals",
    "xor",
    "checkerboard",
    "gaussian",
    "blobs",
    "classification",
)
DATA_PARAMETERS = {
    "n_samples": 1200,
    "noise": 0.2,
    "random_state": 71,
    "n_classes": 4,
    "blob_center_radius": 3.0,
    "blob_cluster_standard_deviation": 1.5,
    "circle_factor": 0.5,
    "spiral_turns": 2,
    "spiral_noise": 0.03,
    "checkerboard_cells": 4,
    "checkerboard_extent": 4.0,
    "anisotropy": 3.0,
    "rotation": 0.35,
    "classification_class_sep": 1.0,
    "classification_flip_y": 0.05,
}
PATH_ESTIMATOR_LABELS = {
    "random_forest": "Random Forest",
    "recursive_partition": "Recursive Partition",
}


def _make_equal_isotropic_gaussians(
    n_samples, mean_distance, standard_deviation, random_state
):
    rng = np.random.default_rng(random_state)
    counts = (n_samples // 2, n_samples - n_samples // 2)
    centers = ((-mean_distance / 2, 0.0), (mean_distance / 2, 0.0))
    covariance = np.eye(2) * standard_deviation**2
    X = np.vstack(
        [
            rng.multivariate_normal(center, covariance, size=count)
            for center, count in zip(centers, counts)
        ]
    )
    y = np.repeat((0, 1), counts)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def _make_blobs(
    n_samples, n_classes, center_radius, cluster_standard_deviation, random_state
):
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_classes), endpoint=False)
    centers = center_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=cluster_standard_deviation,
        random_state=random_state,
    )


def _make_xor(n_samples, cluster_standard_deviation, random_state):
    centers = np.array(
        [(-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)]
    )
    X, cluster_labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_standard_deviation,
        random_state=random_state,
    )
    return X, np.asarray([0, 1, 1, 0])[cluster_labels]


def _make_spirals(n_samples, turns, noise, random_state):
    rng = np.random.default_rng(random_state)
    counts = (n_samples // 2, n_samples - n_samples // 2)
    theta = np.linspace(0.2, 2.0 * np.pi * turns, counts[0])
    radius = np.linspace(0.2, 1.0, counts[0])
    first = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    second = np.column_stack(
        (radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi))
    )
    X = np.vstack((first, second)) + rng.normal(
        scale=noise, size=(n_samples, 2)
    )
    y = np.repeat((0, 1), counts)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def _make_checkerboard(n_samples, cells, extent, label_noise, random_state):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-extent, extent, size=(n_samples, 2))
    cell_indices = np.floor((X + extent) / (2.0 * extent) * cells).astype(int)
    y = (cell_indices[:, 0] + cell_indices[:, 1]) % 2
    flips = rng.random(n_samples) < label_noise
    return X, np.where(flips, 1 - y, y)


def make_2d_dataset(
    dataset="moon",
    *,
    n_samples=600,
    noise=0.24,
    random_state=7,
    mean_distance=3.0,
    standard_deviation=1.0,
    n_classes=3,
    blob_center_radius=3.0,
    blob_cluster_standard_deviation=1.5,
    circle_factor=0.5,
    spiral_turns=1.5,
    spiral_noise=0.12,
    checkerboard_cells=4,
    checkerboard_extent=4.0,
    checkerboard_label_noise=0.0,
    anisotropy=3.0,
    rotation=0.35,
    classification_class_sep=1.0,
    classification_flip_y=0.05,
):
    """Create one of the shared two-dimensional classification datasets."""
    if dataset in ("moon", "moons"):
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if dataset in ("circle", "circles"):
        return make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=circle_factor,
            random_state=random_state,
        )
    if dataset in ("gaussian", "gaussians"):
        return _make_equal_isotropic_gaussians(
            n_samples, mean_distance, standard_deviation, random_state
        )
    if dataset in ("blobs", "blob", "gaussian_blobs"):
        return _make_blobs(
            n_samples,
            n_classes,
            blob_center_radius,
            blob_cluster_standard_deviation,
            random_state,
        )
    if dataset == "xor":
        return _make_xor(n_samples, blob_cluster_standard_deviation, random_state)
    if dataset in ("spiral", "spirals"):
        return _make_spirals(n_samples, spiral_turns, spiral_noise, random_state)
    if dataset in ("checker", "checkerboard"):
        return _make_checkerboard(
            n_samples,
            checkerboard_cells,
            checkerboard_extent,
            checkerboard_label_noise,
            random_state,
        )
    if dataset == "classification":
        return sklearn_make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            class_sep=classification_class_sep,
            flip_y=classification_flip_y,
            random_state=random_state,
        )
    if dataset == "iris":
        iris = load_iris()
        return iris.data[:, :2], iris.target
    if dataset in ("anisotropic", "anisotropic_blobs"):
        X, y = _make_blobs(
            n_samples,
            n_classes,
            blob_center_radius,
            blob_cluster_standard_deviation,
            random_state,
        )
        rotation_matrix = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
        )
        return X @ np.diag((anisotropy, 1.0)) @ rotation_matrix.T, y
    raise ValueError(f"Unknown 2-D dataset: {dataset}")


def plot_probability_heatmap(
    model,
    X,
    y,
    *,
    X_train=None,
    y_train=None,
    padding=0.55,
    grid_size=250,
    title=None,
    ax=None,
    colorbar=True,
    decision_margin=0.1,
):
    """Plot the same probability-weighted class map used by the sister demo."""
    X_points = X if X_train is None else np.asarray(X_train)
    y_points = y if y_train is None else np.asarray(y_train)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, grid_size),
        np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = np.asarray(model.predict_proba(grid), dtype=float).reshape(
        xx.shape + (len(model.classes_),)
    )
    classes = np.asarray(model.classes_)
    palette = np.asarray(
        [plt.get_cmap("tab10")(index % 10)[:3] for index in range(len(classes))]
    )
    predicted_indices = np.argmax(probabilities, axis=2)
    if ax is None:
        fig, axis = plt.subplots(figsize=(9, 6.5))
    else:
        axis, fig = ax, ax.figure

    if len(classes) > 2:
        color_mixture = probabilities @ palette
        safe_probabilities = np.clip(probabilities, np.finfo(float).tiny, 1.0)
        entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=2)
        confidence = 1.0 - entropy / np.log(len(classes))
        rgb = 1.0 - confidence[..., None] * (1.0 - color_mixture)
        axis.imshow(
            np.clip(rgb, 0.0, 1.0),
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            interpolation="nearest",
            aspect="equal",
        )
        axis.contour(
            xx,
            yy,
            predicted_indices,
            levels=np.arange(0.5, len(classes) - 0.5, 1.0),
            linewidths=2,
            colors="black",
        )
        sorted_probabilities = np.sort(probabilities, axis=2)
        margin_surface = sorted_probabilities[:, :, -1] - sorted_probabilities[:, :, -2]
        if margin_surface.min() <= decision_margin <= margin_surface.max():
            axis.contour(
                xx,
                yy,
                margin_surface,
                levels=[decision_margin],
                linewidths=0.45,
                colors="black",
            )
        if colorbar:
            confidence_scale = ScalarMappable(
                norm=Normalize(0.0, 1.0), cmap="Greys_r"
            )
            confidence_scale.set_array(confidence)
            fig.colorbar(confidence_scale, ax=axis).set_label(
                "Confidence (1 − normalized entropy)"
            )
    else:
        probability = probabilities[:, :, 1]
        axis.contourf(
            xx,
            yy,
            probability,
            levels=np.linspace(0, 1, 41),
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            alpha=0.9,
        )
        axis.contour(xx, yy, probability, levels=[0.5], linewidths=2.0, colors="black")
        margin_surface = np.abs(2.0 * probability - 1.0)
        if margin_surface.min() <= decision_margin <= margin_surface.max():
            axis.contour(
                xx,
                yy,
                margin_surface,
                levels=[decision_margin],
                linewidths=0.45,
                colors="black",
            )

    for class_index, class_value in enumerate(classes):
        mask = y_points == class_value
        axis.scatter(
            X_points[mask, 0],
            X_points[mask, 1],
            s=24,
            color=palette[class_index],
            edgecolors="white",
            linewidths=0.45,
            label=f"Class {class_value}",
            zorder=10,
        )
    axis.set_title(title or "Class probability heatmap")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right")
    return fig


def make_path_estimator(
    path_estimator="recursive_partition",
    *,
    n_estimators=100,
    random_state=42,
):
    """Construct the selected internal Recursive Sketch path estimator."""
    if path_estimator == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            n_jobs=-1,
            random_state=random_state,
        )
    if path_estimator == "recursive_partition":
        return BaggedRecursivePartitionClassifier(
            estimator=RecursivePartitionClassifier(
                base_estimator=SVC(kernel="linear", class_weight="balanced")
            ),
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(
        "path_estimator must be one of 'random_forest' or 'recursive_partition'"
    )


def make_sketch(
    path_estimator,
    *,
    n_iterations=2,
    dimension_ratio=20,
    random_state=42,
):
    """Construct the Recursive Sketch transformer around an explicit estimator."""
    return RecursiveSketchClassifier(
        estimator=path_estimator,
        dimension_ratio=dimension_ratio,
        n_iterations=n_iterations,
        dimension_mode="expanding",
        output_format="sparse",
        random_state=random_state,
    )


def make_output_classifier(*, random_state=42):
    """Construct the downstream classifier used for probability estimates."""
    return LogisticRegression(max_iter=3000, random_state=random_state)


def make_model(
    path_estimator="recursive_partition",
    *,
    sketch=None,
    classifier=None,
    n_estimators=100,
    n_iterations=2,
    dimension_ratio=20,
    random_state=42,
):
    """Construct the sketch pipeline from configurable component templates.

    ``sketch`` and ``classifier`` may be supplied as already-instantiated
    estimators. They are cloned before fitting so the same templates can be
    reused safely across datasets.
    """
    if sketch is None:
        sketch = make_sketch(
            make_path_estimator(
                path_estimator,
                n_estimators=n_estimators,
                random_state=random_state,
            ),
            n_iterations=n_iterations,
            dimension_ratio=dimension_ratio,
            random_state=random_state,
        )
    else:
        sketch = clone(sketch)
    if classifier is None:
        classifier = make_output_classifier(random_state=random_state)
    else:
        classifier = clone(classifier)
    return make_pipeline(
        sketch,
        StandardScaler(with_mean=False),
        classifier,
    )


def _path_estimator_label(path_estimator, sketch):
    label = PATH_ESTIMATOR_LABELS[path_estimator]
    if path_estimator == "recursive_partition":
        bagged_estimator = getattr(sketch, "estimator", None)
        base_estimator = getattr(bagged_estimator, "estimator", None)
        base_estimator = getattr(base_estimator, "base_estimator", None)
        kernel = getattr(base_estimator, "kernel", None)
        if kernel:
            return f"{label} + {kernel.upper()} SVM"
    return label


def run_experiment(
    dataset_names=DATASETS,
    *,
    data_parameters=None,
    path_estimator="recursive_partition",
    sketch=None,
    classifier=None,
    n_estimators=100,
    n_iterations=2,
    dimension_ratio=20,
    grid_size=200,
    test_size=0.30,
    random_state=42,
    show=True,
    return_models=False,
):
    """Fit one selected setup on every 2-D dataset and plot its probabilities.

    Explicit ``sketch`` and ``classifier`` templates are cloned per dataset.
    Set ``return_models=True`` to also receive the fitted pipeline for each
    dataset, including its fitted downstream classifier.
    """
    parameters = {**DATA_PARAMETERS, **(data_parameters or {})}
    summary_rows = []
    fitted_models = {}
    for dataset_name in dataset_names:
        X, y = make_2d_dataset(dataset_name, **parameters)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        model = make_model(
            path_estimator,
            sketch=sketch,
            classifier=classifier,
            n_estimators=n_estimators,
            n_iterations=n_iterations,
            dimension_ratio=dimension_ratio,
            random_state=random_state,
        )
        with suppress_sklearn_classification_target_warning():
            model.fit(X_train, y_train)
        fitted_models[dataset_name] = model
        accuracy = model.score(X_test, y_test)
        sketch = model.steps[0][1]
        path_label = _path_estimator_label(path_estimator, sketch)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "path_estimator": path_label,
                "accuracy": accuracy,
                "n_classes": len(model.classes_),
                "output_dimension": sketch.output_dimension_,
            }
        )
        plot_probability_heatmap(
            model,
            X,
            y,
            X_train=X_train,
            y_train=y_train,
            grid_size=grid_size,
            title=(
                f"{dataset_name}\n"
                "Recursive Sketch + Logistic Regression\n"
                f"path estimator: {path_label}\n"
                f"accuracy: {accuracy:.3f}"
            ),
        )
        if show:
            plt.show()
        else:
            plt.close()
    results = pd.DataFrame(summary_rows)
    if return_models:
        return results, fitted_models
    return results


__all__ = [
    "DATASETS",
    "DATA_PARAMETERS",
    "PATH_ESTIMATOR_LABELS",
    "make_2d_dataset",
    "make_model",
    "make_output_classifier",
    "make_path_estimator",
    "make_sketch",
    "plot_probability_heatmap",
    "run_experiment",
]
