"""Reusable experiment and reporting helpers for notebook 16."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from _hypothesis_utils import (
    classification_score,
    make_datasets,
    openml_classification_split,
    suppress_sklearn_classification_target_warning,
    timed_fit_transform,
)
from recursivesketch import RecursiveSketchClassifier

try:
    from recursive_partition import (
        BaggedRecursivePartitionClassifier,
        RecursivePartitionClassifier,
    )
except ModuleNotFoundError:
    search_roots = (Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents)
    sister_repository = next(
        (
            root / 'RecursiveParitionClassifier'
            for root in search_roots
            if (root / 'RecursiveParitionClassifier' / 'recursive_partition').exists()
        ),
        None,
    )
    if sister_repository is None:
        raise ModuleNotFoundError(
            'Install the RecursiveParitionClassifier sister repository or place it next to RecursiveSketch.'
        )
    sys.path.insert(0, str(sister_repository))
    from recursive_partition import (
        BaggedRecursivePartitionClassifier,
        RecursivePartitionClassifier,
    )


def dataset_summary(datasets):
    """Return a compact metadata table for the selected OpenML datasets."""

    return pd.DataFrame([
        {
            'dataset_id': dataset.dataset_id,
            'name': dataset.name,
            'n_original': dataset.n_original,
            'n_used': dataset.n_used,
            'n_features': len(dataset.feature_names),
        }
        for dataset in datasets
    ])


def make_evaluation_blocks(datasets, repetitions, test_size=0.30):
    """Create reproducible train/test blocks shared by every candidate."""

    blocks = []
    for dataset in datasets:
        for repetition_id in repetitions:
            X_train, X_test, y_train, y_test, _ = openml_classification_split(
                dataset,
                test_size=test_size,
                seed=repetition_id,
            )
            blocks.append({
                'dataset_id': dataset.dataset_id,
                'dataset': dataset.name,
                'repetition_id': repetition_id,
                'p_encoded': X_train.shape[1],
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
            })
    return blocks


def make_internal_path_estimator(params, seed, n_jobs=1):
    """Construct one candidate's internal path estimator."""

    if params['path_estimator'] == 'random_forest':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=10,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if params['path_estimator'] == 'recursive_partition':
        return BaggedRecursivePartitionClassifier(
            estimator=RecursivePartitionClassifier(),
            n_estimators=params['n_estimators'],
            n_jobs=n_jobs,
            random_state=seed,
        )
    raise ValueError(f"Unknown path estimator: {params['path_estimator']}")


def make_search_estimator(params, seed, n_jobs=1):
    """Construct a Recursive Sketch candidate from sampled parameters."""

    return RecursiveSketchClassifier(
        estimator=make_internal_path_estimator(params, seed, n_jobs=n_jobs),
        n_components=64,
        n_iterations=params['n_iterations'],
        dimension_ratio=params['dimension_ratio'],
        dimension_mode=params['dimension_mode'],
        output_format='sparse',
        initial_projection_type=params['initial_projection_type'],
        path_projection_type=params['path_projection_type'],
        concat_projection_type=params['concat_projection_type'],
        normalization=params['normalization'],
        random_state=seed,
    )


def make_output_classifier(classifier, seed, n_estimators=100, n_jobs=1):
    """Construct a sparse-compatible downstream classifier."""

    if classifier == 'logistic_regression':
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=10000, random_state=seed),
        )
    if classifier == 'random_forest':
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if classifier == 'svm_rbf_auto_lambda':
        return make_pipeline(
            StandardScaler(with_mean=False),
            SVC(C=1.0, gamma='scale', kernel='rbf', random_state=seed),
        )
    raise ValueError(
        "classifier must be one of 'logistic_regression', 'random_forest', "
        "or 'svm_rbf_auto_lambda'"
    )


def _aligned_tqdm(iterable, progress_description_width='420px', **kwargs):
    progress = tqdm(iterable, **kwargs)
    container = getattr(progress, 'container', None)
    if container is not None:
        description = container.children[0]
        description.layout.width = progress_description_width
        description.layout.min_width = progress_description_width
        description.layout.flex = f'0 0 {progress_description_width}'
    return progress


def run_search(
    *,
    dataset_names,
    dataset_count,
    max_dataset_size,
    dataset_seed,
    repetitions,
    parameter_ranges,
    n_search_iter,
    search_seed,
    test_size=0.30,
    n_jobs=1,
    output_classifier_n_estimators=100,
):
    """Load data, run the random search, and return all reusable results."""

    datasets = make_datasets(
        n=None if dataset_names is not None else dataset_count,
        max_size=max_dataset_size,
        seed=dataset_seed,
        dataset_names=dataset_names,
    )
    evaluation_blocks = make_evaluation_blocks(
        datasets,
        repetitions,
        test_size=test_size,
    )
    sampled_parameters = list(
        ParameterSampler(
            parameter_ranges,
            n_iter=n_search_iter,
            random_state=search_seed,
        )
    )

    search_rows = []
    block_rows = []
    progress_bar_format = (
        '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
        '[{elapsed}<{remaining}, {rate_fmt}]'
    )
    for candidate_id, params in _aligned_tqdm(
        enumerate(sampled_parameters),
        total=len(sampled_parameters),
        desc='Random-search candidates',
        bar_format=progress_bar_format,
    ):
        candidate_start = time.perf_counter()
        candidate_scores = []
        for block in _aligned_tqdm(
            evaluation_blocks,
            desc='Candidate',
            bar_format=progress_bar_format,
            leave=False,
        ):
            model_seed = search_seed + candidate_id * 10000 + block['repetition_id']
            sketch = make_search_estimator(params, seed=model_seed, n_jobs=n_jobs)
            with suppress_sklearn_classification_target_warning():
                X_train_view, fit_wall, fit_cpu = timed_fit_transform(
                    sketch,
                    block['X_train'],
                    block['y_train'],
                )
            accuracy, errors = classification_score(
                make_output_classifier(
                    params['classifier'],
                    seed=model_seed,
                    n_estimators=output_classifier_n_estimators,
                    n_jobs=n_jobs,
                ),
                X_train_view,
                block['y_train'],
                sketch.transform(block['X_test']),
                block['y_test'],
            )
            candidate_scores.append(accuracy)
            block_rows.append({
                'candidate_id': candidate_id,
                'dataset_id': block['dataset_id'],
                'dataset': block['dataset'],
                'repetition_id': block['repetition_id'],
                'p_encoded': block['p_encoded'],
                'accuracy': accuracy,
                'errors': errors,
                'fit_wall_seconds': fit_wall,
                'fit_cpu_seconds': fit_cpu,
            })

        candidate_frame = pd.DataFrame({
            'dataset_id': [block['dataset_id'] for block in evaluation_blocks],
            'dataset': [block['dataset'] for block in evaluation_blocks],
            'accuracy': candidate_scores,
        })
        dataset_means = candidate_frame.groupby(
            ['dataset_id', 'dataset'],
            as_index=False,
        )['accuracy'].mean()
        search_rows.append({
            'candidate_id': candidate_id,
            **params,
            'mean_accuracy': dataset_means['accuracy'].mean(),
            'std_dataset_accuracy': dataset_means['accuracy'].std(ddof=1),
            'mean_block_accuracy': np.mean(candidate_scores),
            'search_seconds': time.perf_counter() - candidate_start,
        })

    search_results = (
        pd.DataFrame(search_rows)
        .sort_values('mean_accuracy', ascending=False)
        .reset_index(drop=True)
    )
    return {
        'datasets': datasets,
        'dataset_summary': dataset_summary(datasets),
        'evaluation_blocks': evaluation_blocks,
        'sampled_parameters': sampled_parameters,
        'search_results': search_results,
        'block_results': pd.DataFrame(block_rows),
    }


def best_configuration(search_results, parameter_ranges, top_selected=1):
    """Return a selected ranked candidate and its parameter dictionary.

    ``top_selected`` is one-based: ``1`` selects the best candidate,
    ``2`` selects the second-best candidate, and so on.
    """

    if (
        isinstance(top_selected, bool)
        or not isinstance(top_selected, (int, np.integer))
        or top_selected < 1
        or top_selected > len(search_results)
    ):
        raise ValueError(
            "top_selected must be a positive rank no greater than the number "
            "of search results"
        )

    selected = search_results.iloc[int(top_selected) - 1]
    best_params = selected[list(parameter_ranges)].to_dict()
    best_candidate_id = int(selected['candidate_id'])
    return best_candidate_id, best_params


def best_scores_by_dataset(block_results, best_candidate_id):
    """Summarize held-out scores for the selected candidate."""

    best_scores = block_results[block_results['candidate_id'] == best_candidate_id]
    return (
        best_scores.groupby(['dataset_id', 'dataset'], as_index=False)
        .agg(
            mean_accuracy=('accuracy', 'mean'),
            std_accuracy=('accuracy', 'std'),
            n_repetitions=('accuracy', 'size'),
        )
        .sort_values('mean_accuracy', ascending=False)
    )


def plot_top_candidates(search_results, top_k=10):
    """Plot the top candidate scores and return the figure."""

    top_results = search_results.head(top_k).copy()
    plot_data = top_results.sort_values('mean_accuracy')
    axis = plot_data.plot(
        x='candidate_id',
        y='mean_accuracy',
        kind='barh',
        xerr='std_dataset_accuracy',
        legend=False,
        figsize=(8, 5),
    )
    axis.set_xlabel('Mean held-out accuracy across datasets')
    axis.set_ylabel('Candidate id')
    axis.set_title('Top random-search candidates')
    figure = axis.get_figure()
    figure.tight_layout()
    return figure


def _parameter_quantiles(search_results, parameter, values):
    grouped = search_results.groupby(parameter)['mean_accuracy']
    quantiles = grouped.quantile([0.25, 0.50, 0.75]).unstack()
    return quantiles.reindex(values)


def plot_marginalized_performance(search_results, parameter_ranges, best_params):
    """Plot the quantile marginal report and return its table and figure."""

    marginal_rows = []
    for parameter, values in parameter_ranges.items():
        for value in values:
            scores = search_results.loc[
                search_results[parameter] == value,
                'mean_accuracy',
            ]
            marginal_rows.append({
                'parameter': parameter,
                'value': value,
                'n_candidates': scores.size,
                'q25_accuracy': scores.quantile(0.25) if scores.size else np.nan,
                'median_accuracy': scores.quantile(0.50) if scores.size else np.nan,
                'q75_accuracy': scores.quantile(0.75) if scores.size else np.nan,
            })
    marginal_summary = pd.DataFrame(marginal_rows)

    n_columns = 3
    n_rows = int(np.ceil(len(parameter_ranges) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, 4.5 * n_rows),
        squeeze=False,
    )
    for axis, (parameter, allowed_values) in zip(
        axes.flat,
        parameter_ranges.items(),
    ):
        values = list(allowed_values)
        quantiles = _parameter_quantiles(search_results, parameter, values)
        positions = np.arange(len(values))
        observed = quantiles[0.50].notna().to_numpy()
        if observed.any():
            axis.fill_between(
                positions[observed],
                quantiles.loc[observed, 0.25].to_numpy(),
                quantiles.loc[observed, 0.75].to_numpy(),
                color='#9ecae1',
                alpha=0.7,
                label='25th–75th percentile',
            )
            axis.plot(
                positions[observed],
                quantiles.loc[observed, 0.50].to_numpy(),
                marker='o',
                color='#08519c',
                linewidth=2,
                label='Median',
            )
        for position in positions[~observed]:
            axis.axvline(position, color='0.65', linestyle='--', linewidth=0.8)
        best_value = best_params[parameter]
        if best_value in values:
            axis.axvline(
                values.index(best_value),
                color='red',
                linestyle='-',
                linewidth=1.0,
                label='Best configuration',
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [str(value) for value in values],
            rotation=35,
            ha='right',
        )
        axis.set_title(parameter)
        axis.set_ylabel('Dataset-averaged accuracy')
        axis.grid(axis='y', alpha=0.25)
    for axis in axes.flat[len(parameter_ranges):]:
        axis.set_visible(False)
    figure.suptitle('Marginalized random-search performance', y=1.02)
    figure.tight_layout()
    return marginal_summary, figure


def _value_position(values, value):
    for position, candidate in enumerate(values):
        if candidate == value:
            return position
    return None


def plot_pairwise_performance(search_results, parameter_ranges, best_params):
    """Plot upper-triangular pairwise performance with diagonal marginals."""

    if search_results.empty:
        raise ValueError('search_results is empty; run the search before plotting pairwise results.')

    parameters = list(parameter_ranges)
    n_parameters = len(parameters)
    figure_size = max(12, 2.8 * n_parameters)
    panel_margin = 0.12
    figure = plt.figure(figsize=(figure_size + 1, figure_size))
    figure.subplots_adjust(
        left=0.04,
        right=0.97,
        bottom=0.04,
        top=0.96,
    )
    grid = figure.add_gridspec(
        n_parameters,
        n_parameters + 2,
        width_ratios=[1] * n_parameters + [0.5, 0.12],
        wspace=0.25,
        hspace=0.25,
    )
    axes = np.empty((n_parameters, n_parameters), dtype=object)
    for row in range(n_parameters):
        for column in range(n_parameters):
            sharex = axes[0, column] if row else None
            axes[row, column] = figure.add_subplot(
                grid[row, column],
                sharex=sharex,
            )
    colorbar_axis = figure.add_subplot(grid[:, -1])
    score = search_results['mean_accuracy']
    score_norm = plt.Normalize(vmin=score.min(), vmax=score.max())
    iso_levels = [
        (0.50, score.quantile(0.50), '#08519c'),
        (0.75, score.quantile(0.75), '#e6550d'),
    ]

    for row, y_parameter in enumerate(parameters):
        y_values = list(parameter_ranges[y_parameter])
        for column, x_parameter in enumerate(parameters):
            axis = axes[row, column]
            if row > column:
                axis.set_visible(False)
                continue

            x_values = list(parameter_ranges[x_parameter])
            x_positions = np.arange(len(x_values))
            y_positions = np.arange(len(y_values))
            best_x = _value_position(x_values, best_params[x_parameter])
            best_y = _value_position(y_values, best_params[y_parameter])

            if row == column:
                quantiles = _parameter_quantiles(
                    search_results,
                    y_parameter,
                    y_values,
                )
                observed = quantiles[0.50].notna().to_numpy()
                if observed.any():
                    axis.fill_between(
                        x_positions[observed],
                        quantiles.loc[observed, 0.25].to_numpy(),
                        quantiles.loc[observed, 0.75].to_numpy(),
                        color='#9ecae1',
                        alpha=0.7,
                    )
                    axis.plot(
                        x_positions[observed],
                        quantiles.loc[observed, 0.50].to_numpy(),
                        marker='o',
                        color='#08519c',
                        linewidth=2,
                    )
                for position in x_positions[~observed]:
                    axis.axvline(position, color='0.65', linestyle='--', linewidth=0.8)
                if best_x is not None:
                    axis.axvline(best_x, color='red', linewidth=1.0)
                if row == 0:
                    axis.set_ylabel('Dataset-averaged accuracy')
            else:
                pair_scores = (
                    search_results.groupby(
                        [x_parameter, y_parameter],
                        as_index=False,
                    )['mean_accuracy'].mean()
                )
                pair_x = pair_scores[x_parameter].map(
                    {value: position for position, value in enumerate(x_values)}
                ).to_numpy(dtype=float)
                pair_y = pair_scores[y_parameter].map(
                    {value: position for position, value in enumerate(y_values)}
                ).to_numpy(dtype=float)
                pair_z = pair_scores['mean_accuracy'].to_numpy(dtype=float)
                axis.scatter(
                    pair_x,
                    pair_y,
                    c=pair_z,
                    cmap='Blues_r',
                    norm=score_norm,
                    s=20,
                    alpha=0.65,
                    edgecolors='none',
                )
                unique_points = len(np.unique(np.column_stack([pair_x, pair_y]), axis=0))
                if len(pair_z) >= 3 and unique_points >= 3:
                    try:
                        x_bounds = (
                            float(x_positions[0] - panel_margin),
                            float(x_positions[-1] + panel_margin),
                        )
                        y_bounds = (
                            float(y_positions[0] - panel_margin),
                            float(y_positions[-1] + panel_margin),
                        )
                        boundary_points = [
                            (x_bounds[0], y_bounds[0]),
                            (x_bounds[0], y_bounds[1]),
                            (x_bounds[1], y_bounds[0]),
                            (x_bounds[1], y_bounds[1]),
                        ]
                        observed_points = set(zip(pair_x, pair_y))
                        added_points = [
                            point
                            for point in boundary_points
                            if point not in observed_points
                        ]
                        contour_x = pair_x
                        contour_y = pair_y
                        contour_z = pair_z
                        if added_points:
                            added_values = np.array([
                                pair_z[np.argmin(
                                    (pair_x - point[0]) ** 2
                                    + (pair_y - point[1]) ** 2
                                )]
                                for point in added_points
                            ])
                            contour_x = np.concatenate([
                                pair_x,
                                [point[0] for point in added_points],
                            ])
                            contour_y = np.concatenate([
                                pair_y,
                                [point[1] for point in added_points],
                            ])
                            contour_z = np.concatenate([pair_z, added_values])

                        triangulation = mtri.Triangulation(
                            contour_x,
                            contour_y,
                        )
                        if contour_z.min() < contour_z.max():
                            axis.tricontourf(
                                triangulation,
                                contour_z,
                                levels=np.linspace(
                                    contour_z.min(), contour_z.max(), 8
                                ),
                                cmap='Blues_r',
                                norm=score_norm,
                                alpha=0.55,
                            )
                        drawn_levels = set()
                        for quantile, level, colour in iso_levels:
                            if level in drawn_levels or not (pair_z.min() <= level <= pair_z.max()):
                                continue
                            contours = axis.tricontour(
                                triangulation,
                                contour_z,
                                levels=[level],
                                colors=[colour],
                                linewidths=1.5,
                            )
                            axis.clabel(
                                contours,
                                inline=True,
                                fontsize=7,
                                fmt={level: f'{int(100 * quantile)}%'},
                            )
                            drawn_levels.add(level)
                    except (RuntimeError, ValueError):
                        pass
                if best_x is not None and best_y is not None:
                    axis.plot(
                        best_x,
                        best_y,
                        marker='o',
                        markersize=10,
                        markerfacecolor='none',
                        markeredgecolor='red',
                        markeredgewidth=2,
                        linestyle='None',
                    )

            axis.set_xticks(x_positions)
            axis.set_xticklabels(
                [str(value) for value in x_values],
                rotation=45,
                ha='right',
                fontsize=8,
            )
            axis.tick_params(
                axis='x',
                labelbottom=row == column,
            )
            if row != column:
                axis.set_yticks(y_positions)
                if column == n_parameters - 1:
                    axis.yaxis.tick_right()
                    axis.set_yticklabels(
                        [str(value) for value in y_values],
                        fontsize=8,
                    )
                    axis.tick_params(axis='y', labelleft=False, labelright=True)
                else:
                    axis.tick_params(axis='y', labelleft=False, labelright=False)
            else:
                axis.tick_params(axis='y', labelleft=row == 0)
            axis.set_xlim(
                x_positions[0] - panel_margin,
                x_positions[-1] + panel_margin,
            )
            if row != column:
                axis.set_ylim(
                    y_positions[0] - panel_margin,
                    y_positions[-1] + panel_margin,
                )
            else:
                axis.margins(y=0.08)
            if row == 0:
                axis.set_xlabel(x_parameter)
                axis.xaxis.set_label_position('top')
                axis.xaxis.labelpad = 8
            else:
                axis.set_xlabel('')
            axis.set_title(y_parameter if row == column and row != 0 else '')
            axis.grid(alpha=0.2)
            axis.set_box_aspect(1)

    mappable = plt.cm.ScalarMappable(norm=score_norm, cmap='Blues_r')
    mappable.set_array([])
    figure.colorbar(
        mappable,
        cax=colorbar_axis,
        label='Dataset-averaged accuracy',
    )
    figure.suptitle('Pairwise hyperparameter performance', y=0.995)
    return figure


__all__ = [
    'best_configuration',
    'best_scores_by_dataset',
    'dataset_summary',
    'make_evaluation_blocks',
    'make_internal_path_estimator',
    'make_output_classifier',
    'make_search_estimator',
    'plot_marginalized_performance',
    'plot_pairwise_performance',
    'plot_top_candidates',
    'run_search',
]
