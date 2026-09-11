from typing import Any, Callable, Iterable, Literal

import mlflow
import random
import hashlib
from itertools import combinations, chain
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline

Step = tuple[str, TransformerMixin]
ColumnTransformerStep = tuple[str, TransformerMixin, Iterable[str] | Callable]

_METRICS = {
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "r2": r2_score,
}


class ColumnTransformerBuilder:
    def __init__(
        self,
        *,
        name: str = "",
        transformers: Iterable[ColumnTransformerStep] = (),
        banned: Iterable[str] = (),
        chained: tuple[
            str | None, str | Iterable[str] | Callable | None, Iterable[Step]
        ] = (
            None,
            None,
            (),
        ),
    ) -> None:
        self.transformers: list[ColumnTransformerStep] = []
        for n, e, s in transformers:
            if callable(s):
                inner = select
                select = lambda X: [c for c in inner(X) if c not in banned]
            else:
                select = list(s)
            self.transformers.append((n, clone(e), select))

        self.name: str = name
        self.banned: list[str] = list(banned)
        chain_name, chained_select, chain = chained
        if (
            not callable(chained_select)
            and not isinstance(chained_select, str)
            and chained_select is not None
        ):
            chained_select = list(chained_select)

        self.chained: tuple[
            str | None, str | Iterable[str] | Callable | None, list[Step]
        ] = (
            chain_name,
            chained_select,
            [(n, clone(s)) for n, s in chain],
        )

    def _evolve(self, **changes: Any) -> "ColumnTransformerBuilder":
        return ColumnTransformerBuilder(
            transformers=changes.get("transformers", self.transformers),
            name=changes.get("name", self.name),
            chained=changes.get("chained", self.chained),
        )

    def with_transformer(
        self,
        transformer: TransformerMixin,
        select: Iterable[str] | str | Callable,
        *,
        name: str | None = None,
        chain_name: str | None = None,
    ) -> "ColumnTransformerBuilder":
        next = self._finish_chained()
        if name is None:
            name = transformer.__class__.__name__
        assert select is not None, (
            "When building a column transformer, columns must be selected"
        )

        if callable(select):
            inner, banned = select, tuple(self.banned)
            select = lambda X: [c for c in inner(X) if c not in banned]
        else:
            if isinstance(select, str):
                select = [select]
            select = [s for s in select if s not in self.banned]
        step: ColumnTransformerStep = (name, transformer, select)

        steps = list(self.transformers)
        if chain_name is None:
            steps.append(step)
        return next._evolve(
            transformers=steps, chained=(chain_name, select, [(name, transformer)])
        )

    def _finish_chained(self) -> "ColumnTransformerBuilder":
        name, select, steps = self.chained
        if name is None:
            rand = hashlib.sha1(str(random.getrandbits(128)).encode()).hexdigest()[:6]
            name = f"{Pipeline.__class__.__name__}_{rand}"

        if len(steps) > 1:
            pipeline = Pipeline([*steps])
            step = (name, pipeline, select)
            steps = list(self.transformers)
            steps.append(step)
            return self._evolve(transformers=steps, chained=(None, None, ()))
        return self

    def with_chained(
        self,
        transformer: TransformerMixin,
        *,
        name: str | None = None,
    ) -> "ColumnTransformerBuilder":
        if name is None:
            name = transformer.__class__.__name__

        chain_name, select, steps = self.chained
        step: Step = (name, transformer)

        steps = list(steps)
        steps.append(step)
        return self._evolve(chained=(chain_name, select, steps))

    def build(self) -> Step:
        end = self._finish_chained()
        return (
            self.name,
            ColumnTransformer(
                end.transformers,
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        )


class PipelineBuilder:
    def __init__(
        self,
        *,
        transformers: Iterable[Step] = (),
        pinned: Iterable[bool] = (),
        banned: Iterable[str] = (),
        regressor: RegressorMixin | None = None,
        regressor_name: str | None = None,
    ) -> None:
        self.transformers: list[Step] = [(n, clone(e)) for n, e in transformers]
        self.pinned: list[bool] = list(pinned)
        self.banned: list[str] = list(banned)
        self.regressor: RegressorMixin | None = (
            clone(regressor) if regressor is not None else None
        )
        self.regressor_name: str | None = regressor_name

    def _evolve(self, **changes: Any) -> "PipelineBuilder":
        return PipelineBuilder(
            transformers=changes.get("transformers", self.transformers),
            pinned=changes.get("pinned", self.pinned),
            banned=changes.get("banned", self.banned),
            regressor=changes.get("regressor", self.regressor),
            regressor_name=changes.get("regressor_name", self.regressor_name),
        )

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

    def with_transformer(
        self,
        transformer: TransformerMixin,
        *,
        pin: bool = False,
        name: str | None = None,
        before: Any | None = None,
        select: Iterable[str] | str | Callable | None = None,
    ) -> "PipelineBuilder":
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

    def with_column_transformer(
        self,
        builder: Callable[["ColumnTransformerBuilder"], "ColumnTransformerBuilder"],
        *,
        pin: bool = False,
        name: str | None = None,
        before: Any | None = None,
    ) -> "PipelineBuilder":
        if name is None:
            rand = hashlib.sha1(str(random.getrandbits(128)).encode()).hexdigest()[:6]
            name = f"{ColumnTransformer.__class__.__name__}_{rand}"

        step = builder(ColumnTransformerBuilder(name=name, banned=self.banned)).build()
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

    def without_transformer(self, transformer) -> "PipelineBuilder":
        steps = list(self.transformers)
        pinned = list(self.pinned)
        at = self._find_before(steps, transformer)
        assert at is not None, f"Could not find transformer to remove: {transformer}"
        fst, snd = steps[: at - 1], steps[at:]
        pinned_fst, pinned_snd = pinned[: at - 1], pinned[at:]
        return self._evolve(transformers=fst + snd, pinned=pinned_fst + pinned_snd)

    def with_regressor(
        self, regressor: RegressorMixin, *, name: str | None = None
    ) -> "PipelineBuilder":
        if name is None:
            name = type(regressor).__name__
        return self._evolve(regressor=regressor, regressor_name=name)

    def build(self) -> Pipeline:
        steps = list(self.transformers)
        if self.regressor is not None:
            steps.append((self.regressor_name, self.regressor))
        return clone(Pipeline(steps)).set_output(transform="polars")


class ExperimentBuilder:
    def __init__(
        self,
        label: str,
        *,
        pipeline: PipelineBuilder | None = None,
        regressors: Iterable[RegressorMixin] = (),
        features: Iterable[str] = (),
        banned: Iterable[str] = (),
        experiment_name: str = "",
        run_name: str = "",
    ) -> None:
        self.did_run: bool = False
        self.label: str = label
        self.banned: list[str] = list(banned)
        self.pipeline: PipelineBuilder = (
            pipeline if pipeline is not None else PipelineBuilder(banned=self.banned)
        )
        self.regressors: list[RegressorMixin] = [clone(r) for r in regressors]
        self.features: list[str] = list(features)
        self._experiment_name: str = experiment_name
        self._run_name: str = run_name

    def _evolve(self, **changes: Any) -> "ExperimentBuilder":
        return ExperimentBuilder(
            changes.get("label", self.label),
            pipeline=changes.get("pipeline", self.pipeline),
            regressors=changes.get("regressors", self.regressors),
            features=changes.get("features", self.features),
            banned=changes.get("banned", self.banned),
            experiment_name=changes.get("experiment_name", self._experiment_name),
            run_name=changes.get("run_name", self._run_name),
        )

    def _pruned(self, features: Iterable[str]) -> list[str]:
        return [f for f in features if f != self.label]

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

    def with_pipeline(
        self, builder: Callable[[PipelineBuilder], PipelineBuilder]
    ) -> "ExperimentBuilder":
        pipeline = builder(self.pipeline._evolve(banned=self.banned))
        return self._evolve(pipeline=pipeline)

    def build_transformers(self) -> Pipeline:
        return self.pipeline.build()

    def pipeline_for(self, regressor: RegressorMixin) -> Pipeline:
        return self.pipeline.with_regressor(regressor).build()

    # copied from: https://docs.python.org/2/library/itertools.html#recipes
    @staticmethod
    def _powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    @staticmethod
    def _run_worker(
        builder: "ExperimentBuilder",
        df: pl.DataFrame,
        tracking_uri: str,
        log_model: bool,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        if log_model:
            mlflow.sklearn.autolog()
        builder.run_experiment(df, log_model=log_model)

    def exhaustive(
        self,
        df: pl.DataFrame,
        *,
        select: str | Iterable[str] | None = None,
        conflicting: dict[str, str | Iterable[str]] = {},
        keep_on_present: dict[str, str | Iterable[str]] = {},
        n_jobs: int = -1,
        log_model: bool = False,
    ) -> None:
        """
        Exhaustive experiment runs with different transformer combinations
        This is an expensive operation only 5 transformers results in 32 experiment runs!
        """
        all_names = {name for name, _ in self.pipeline.transformers}
        for one, others in conflicting.items():
            others = {others} if isinstance(others, str) else set(others)
            for name in {one, *others}:
                assert name in all_names, (
                    f"conflicting references unknown transformer {name!r}"
                )
        for name in keep_on_present:
            assert name in all_names, (
                f"keep_on_present references unknown transformer {name!r}"
            )

        not_pinned = [
            (idx, step)
            for idx, (step, pinned) in enumerate(
                zip(self.pipeline.transformers, self.pipeline.pinned)
            )
            if not pinned
        ]
        pinned = [
            (idx, step)
            for idx, (step, pinned) in enumerate(
                zip(self.pipeline.transformers, self.pipeline.pinned)
            )
            if pinned
        ]
        if select is None:
            base_select: set[str] = set()
        elif isinstance(select, str):
            base_select = {select}
        else:
            base_select = set(select)

        builders: list["ExperimentBuilder"] = []
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

            using_names: set[str] = set(map(lambda x: x[0], transformers))
            conflict = False
            for one, others in conflicting.items():
                if conflict:
                    break
                if isinstance(others, str):
                    others = [others]
                others = set(others)
                for other in others:
                    if len(using_names & set((one, other))) > 1:
                        conflict = True
                        break

            if conflict:
                continue

            if len(using_names) == 0:
                run_name = "no transformers"
            else:
                run_name = f"using: {','.join(using_names)}"

            current_select = set(base_select)
            for keep_name, keep_cols in keep_on_present.items():
                if keep_name in using_names:
                    if isinstance(keep_cols, str):
                        keep_cols = [keep_cols]
                    current_select |= set(keep_cols)

            builders.append(
                self._evolve(
                    pipeline=self.pipeline._evolve(transformers=transformers),
                    run_name=run_name,
                ).select(*current_select)
            )
        _ = list(
            tqdm(
                Parallel(
                    n_jobs=n_jobs, backend="loky", return_as="generator_unordered"
                )(
                    delayed(self._run_worker)(
                        b, df, mlflow.get_tracking_uri(), log_model
                    )
                    for b in builders
                ),
                total=len(builders),
            )
        )

    def run_experiment(
        self,
        df: pl.DataFrame,
        regressor: RegressorMixin | None = None,
        validation: Literal["cv", "train_test"] = "cv",
        log_model: bool = True,
    ) -> None:
        if self.did_run:
            return
        if self._experiment_name:
            mlflow.set_experiment(self._experiment_name)
        if regressor is None:
            for regressor in self.regressors:
                self._run_one(df, regressor, validation=validation, log_model=log_model)
        else:
            self._run_one(df, regressor, validation=validation, log_model=log_model)
        self.did_run = True

    def _already_logged(self, run_name: str) -> str | None:
        experiment_id = mlflow.tracking.fluent._get_experiment_id()
        runs = list(
            mlflow.search_runs(
                experiment_ids=[experiment_id],
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

    def _run_one(
        self,
        df: pl.DataFrame,
        regressor: RegressorMixin,
        validation: Literal["cv", "train_test"] = "cv",
        log_model: bool = True,
    ) -> None:
        run_name = self._run_name_for(regressor)
        if run_url := self._already_logged(self._run_name_for(regressor)):
            print(f"{run_name} already run: {run_url}")
            return
        with mlflow.start_run(run_name=run_name):
            if validation == "cv":
                metrics = self._cv_scores(df, regressor, log_folds=True)
            else:
                metrics = self._train_test_scores(df, regressor)
            mlflow.log_metrics(metrics)
            final = self._fit_full(df, regressor)
            self._log_dataset(df, final)
            if log_model:
                info = mlflow.sklearn.log_model(
                    final,
                    name="model",
                    serialization_format="cloudpickle",
                    pip_requirements=["scikit-learn", "polars", "numpy"],
                )
                mlflow.log_metrics(metrics, model_id=info.model_id)

    def _fit_full(self, df: pl.DataFrame, regressor: RegressorMixin) -> Pipeline:
        pipeline = self.pipeline_for(regressor)
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

    def _log_pred_plot(
        self, y_true: pl.DataFrame, y_pred: np.ndarray, artifact_file: str
    ) -> None:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        fig, ax = plt.subplots()
        index = np.arange(len(y_true))
        ax.plot(index, y_true, label="actual")
        ax.plot(index, y_pred, label="predicted")
        ax.set_xlabel("index")
        ax.set_ylabel(self.label)
        ax.legend()
        mlflow.log_figure(fig, artifact_file)
        plt.close(fig)

    def _train_test_scores(
        self, df: pl.DataFrame, regressor: RegressorMixin
    ) -> dict[str, float]:
        pipeline = self.pipeline_for(regressor)
        X = df.select(self.features)
        y = df.select([self.label, "real_obs_weather", "real_obs_power"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

        y_train = y_train.select(self.label)
        pipeline.fit(X_train, y_train)

        mask = y_test["real_obs_weather"]
        X_test, y_test = (
            X_test.filter(mask),
            y_test.filter(mask).select(pl.col("real_obs_power").alias(self.label)),
        )

        y_pred = pipeline.predict(X_test)
        self._log_pred_plot(y_test, y_pred, "train_test_predictions.png")
        return {name: fn(y_test, y_pred) for name, fn in _METRICS.items()}

    def _cv_scores(
        self,
        df: pl.DataFrame,
        regressor: RegressorMixin,
        *,
        log_folds: bool = False,
    ) -> dict[str, float]:
        cv = TimeSeriesSplit(n_splits=5)
        pipeline = self.pipeline_for(regressor)
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
                    self._log_pred_plot(y_test, y_pred, "fold_predictions.png")

        summary: dict[str, float] = {}
        for name, values in acc.items():
            lo, hi = min(values), max(values)
            summary[f"best_{name}"] = hi if name == "r2" else lo
            summary[f"worst_{name}"] = lo if name == "r2" else hi
            summary[f"mean_{name}"] = float(np.mean(values))
        return summary
