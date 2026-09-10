from typing import Any, Callable, Iterable

import mlflow
from itertools import combinations, chain
from collections import deque
import numpy as np
import polars as pl
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

Step = tuple[str, TransformerMixin]

_METRICS = {
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "r2": r2_score,
}


class ExperimentBuilder:
    def __init__(
        self,
        label: str,
        *,
        pinned: Iterable[bool] = (),
        transformers: Iterable[Step] = (),
        regressors: Iterable[RegressorMixin] = (),
        features: Iterable[str] = (),
        banned: Iterable[str] = (),
        experiment_name: str = "",
        run_name: str = "",
    ) -> None:
        self.did_run: bool = False
        self.label: str = label
        self.pinned: list[bool] = list(pinned)
        self.transformers: list[Step] = [(n, clone(e)) for n, e in transformers]
        self.regressors: list[RegressorMixin] = [clone(r) for r in regressors]
        self.features: list[str] = list(features)
        self.banned: list[str] = list(banned)
        self._experiment_name: str = experiment_name
        self._run_name: str = run_name

    def _evolve(self, **changes: Any) -> "ExperimentBuilder":
        return ExperimentBuilder(
            changes.get("label", self.label),
            transformers=changes.get("transformers", self.transformers),
            regressors=changes.get("regressors", self.regressors),
            features=changes.get("features", self.features),
            banned=changes.get("banned", self.banned),
            experiment_name=changes.get("experiment_name", self._experiment_name),
            run_name=changes.get("run_name", self._run_name),
            pinned=changes.get("pinned", self.pinned),
        )

    def _pruned(self, features: Iterable[str]) -> list[str]:
        return [f for f in features if f != self.label]

    @staticmethod
    def _find_before(steps: list[Step], before: Any | None) -> int | None:
        if before is None:
            return None
        is_str = isinstance(before, str)
        typ = type(before)
        for i, (nm, tr) in enumerate(steps):
            if (is_str and nm == before) or (not is_str and isinstance(tr, typ)):
                return i + 1
            if isinstance(tr, ColumnTransformer) and tr.transformers:
                for inner_nm, inner_tr, _ in tr.transformers:
                    if is_str and inner_nm == before:
                        return i + 1
                    if not is_str and isinstance(inner_tr, typ):
                        return i + 1
        return None

    def _run_name_for(self, regressor: RegressorMixin) -> str:
        base = type(regressor).__name__
        return f"{base}: {self._run_name}" if self._run_name else base

    def with_experiment_name(self, name: str) -> "ExperimentBuilder":
        return self._evolve(experiment_name=name)

    def with_run_name(self, name: str) -> "ExperimentBuilder":
        return self._evolve(run_name=name)

    def using_regressors(self, *regressors: RegressorMixin) -> "ExperimentBuilder":
        if len(regressors) == 1 and isinstance(regressors[0], (list, tuple)):
            regressors = tuple(regressors[0])
        return self._evolve(regressors=regressors)

    def with_features(self, *features: str) -> "ExperimentBuilder":
        return self._evolve(features=list(features))

    def select(self, *features: str) -> "ExperimentBuilder":
        return self._evolve(features=self._pruned(features))

    def drop(self, *features: str) -> "ExperimentBuilder":
        return self._evolve(
            features=self._pruned(f for f in self.features if f not in features)
        )

    def ignore(self, *features: str) -> "ExperimentBuilder":
        return self._evolve(banned=[*self.banned, *features])

    def add(self, *features: str) -> "ExperimentBuilder":
        return self._evolve(features=self._pruned([*self.features, *features]))

    def with_transformer(
        self,
        *,
        pin: bool = False,
        name: str | None = None,
        transformer: TransformerMixin,
        before: Any | None = None,
        select: Iterable[str] | str | Callable | None = None,
    ) -> "ExperimentBuilder":
        if name is None:
            name = transformer.__class__.__name__

        if select is not None:
            if callable(select):
                inner, banned = select, tuple(self.banned)
                select = lambda X: [c for c in inner(X) if c not in banned]
            else:
                if isinstance(select, str):
                    select = [select]
                select = [s for s in select if s not in self.banned]
            step: Step = (
                name,
                ColumnTransformer(
                    [(name, transformer, select)],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            )
        else:
            step = (name, transformer)

        steps = list(self.transformers)
        pinned = list(self.pinned)
        at = self._find_before(steps, before)
        if at is None:
            pinned.append(pin)
            steps.append(step)
        else:
            pinned.insert(at, pin)
            steps.insert(at, step)
        return self._evolve(transformers=steps, pinned=pinned)

    def without_transformer(self, transformer) -> "ExperimentBuilder":
        steps = list(self.transformers)
        pinned = list(self.pinned)
        at = self._find_before(steps, transformer)
        assert at is not None, f"Could not find transformer to remove: {transformer}"
        fst, snd = steps[: at - 1], steps[at:]
        pinned_fst, pinned_snd = pinned[: at - 1], pinned[at:]
        return self._evolve(transformers=fst + snd, pinned=pinned_fst + pinned_snd)

    def build_transformers(self) -> Pipeline:
        steps = list(self.transformers)
        return clone(Pipeline(steps)).set_output(transform="polars")

    def _pipeline_for(self, regressor: RegressorMixin) -> Pipeline:
        steps = [*self.transformers, (type(regressor).__name__, regressor)]
        return clone(Pipeline(steps)).set_output(transform="polars")

    # copied from: https://docs.python.org/2/library/itertools.html#recipes
    @staticmethod
    def _powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def exhaustive(
        self, df: pl.DataFrame, drop_on_missing: dict[str, Iterable[str]] = {}
    ) -> None:
        """
        Exhaustive experiment runs with different transformer combinations
        This is an expensive operation only 5 transformers results in 32 experiment runs!
        """
        not_pinned = [
            (idx, step)
            for idx, (step, pinned) in enumerate(zip(self.transformers, self.pinned))
            if not pinned
        ]
        pinned = [
            (idx, step)
            for idx, (step, pinned) in enumerate(zip(self.transformers, self.pinned))
            if pinned
        ]
        for subset in self._powerset(not_pinned):
            # a stack could be used but I cannot bother
            q: deque[tuple[int, Step]] = deque(subset)
            transformers: list[Step] = []
            pinned_idx = 0
            while q:
                idx, step = q.popleft()
                if pinned_idx < len(pinned) and pinned[pinned_idx][0] < idx:
                    transformers.append(pinned[pinned_idx][1])
                    q.appendleft((idx, step))
                    pinned_idx += 1
                    continue
                transformers.append(step)
            while pinned_idx < len(pinned):
                transformers.append(pinned[pinned_idx][1])
                pinned_idx += 1

            using_names: list[str] = list(map(lambda x: x[0], transformers))
            if len(using_names) == 0:
                run_name = "no transformers"
            else:
                run_name = f"using: {','.join(using_names)}"

            builder = self._evolve(transformers=transformers, run_name=run_name)
            for drop_name, drop_cols in drop_on_missing.items():
                if drop_name not in using_names:
                    cols = (drop_cols,) if isinstance(drop_cols, str) else drop_cols
                    builder = builder.drop(*cols)

            builder.run_experiment(df)

    def run_experiment(self, df: pl.DataFrame) -> None:
        if self.did_run:
            return
        assert self._experiment_name, (
            "To run an experiment you have to provide an experiment name; use with_experiment_name"
        )
        mlflow.set_experiment(self._experiment_name)
        for regressor in self.regressors:
            self._run_one(df, regressor)
        self.did_run = True

    def _already_logged(self, run_name: str) -> str | None:
        runs = list(
            mlflow.search_runs(
                experiment_names=[self._experiment_name],
                filter_string=(
                    f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'"
                ),
                output_format="list",
                max_results=1,
            )
        )
        if len(runs) > 0:
            base = mlflow.get_tracking_uri()
            run_info = runs[0].info
            experiment = run_info.experiment_id
            run_id = run_info.run_id
            return f"{base}/#/experiments/{experiment}/runs/{run_id}"
        return None

    def _run_one(self, df: pl.DataFrame, regressor: RegressorMixin) -> None:
        run_name = self._run_name_for(regressor)
        if run_url := self._already_logged(self._run_name_for(regressor)):
            print(f"{run_name} already run: {run_url}")
            return
        with mlflow.start_run(run_name=run_name):
            metrics = self._cv_scores(df, regressor, log_folds=True)
            mlflow.log_metrics(metrics)
            final = self._fit_full(df, regressor)
            self._log_dataset(df, final)
            info = mlflow.sklearn.log_model(
                final, name="model", serialization_format="cloudpickle"
            )
            mlflow.log_metrics(metrics, model_id=info.model_id)

    def _fit_full(self, df: pl.DataFrame, regressor: RegressorMixin) -> Pipeline:
        pipeline = self._pipeline_for(regressor)
        pipeline.fit(df.select(self.features), df.select(self.label))
        return pipeline

    def _log_dataset(self, df: pl.DataFrame, pipeline: Pipeline) -> None:
        frame = df.select([*self.features, self.label])
        y_pred = np.asarray(pipeline.predict(df.select(self.features))).ravel()
        frame = frame.with_columns(pl.Series("prediction", y_pred))
        dataset = mlflow.data.from_polars(
            frame,
            targets=self.label,
            predictions="prediction",
            name="training",
        )
        mlflow.log_input(dataset, context="training")

    def _cv_scores(
        self,
        df: pl.DataFrame,
        regressor: RegressorMixin,
        *,
        log_folds: bool = False,
    ) -> dict[str, float]:
        cv = TimeSeriesSplit(n_splits=5)
        pipeline = self._pipeline_for(regressor)
        acc: dict[str, list[float]] = {name: [] for name in _METRICS}
        for i, (train_index, test_index) in enumerate(cv.split(df)):
            fold = clone(pipeline)
            df_train = df[train_index]
            df_test = df[test_index].filter(pl.col("real_obs_weather"))
            X_train = df_train.select(self.features)
            X_test = df_test.select(self.features)
            y_test = df_test.select(pl.col("real_obs_power").alias(self.label))

            fold.fit(X_train, df_train.select(self.label))
            y_pred = fold.predict(X_test)
            scores = {name: fn(y_test, y_pred) for name, fn in _METRICS.items()}
            for name, value in scores.items():
                acc[name].append(value)

            if log_folds:
                with mlflow.start_run(run_name=f"Fold {i}", nested=True):
                    mlflow.log_metrics(
                        {
                            **scores,
                            "train_size": X_train.shape[0],
                            "test_size": X_test.shape[0],
                        }
                    )

        summary: dict[str, float] = {}
        for name, values in acc.items():
            lo, hi = min(values), max(values)
            summary[f"best_{name}"] = hi if name == "r2" else lo
            summary[f"worst_{name}"] = lo if name == "r2" else hi
            summary[f"mean_{name}"] = float(np.mean(values))
        return summary
