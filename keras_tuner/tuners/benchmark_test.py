# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import keras_tuner
from keras_tuner.engine import hyperparameters as hp_module


@pytest.mark.benchmark(
    group="Float Optimization Benchmark of TensorFlow Einsum",
    min_rounds=10,
    max_time=6000,
)
def test_float_optimization_benchmark_tf(tmp_path, benchmark):
    class PolynomialTuner(keras_tuner.engine.base_tuner.BaseTuner):
        def run_trial(self, trial):
            hp = trial.hyperparameters
            return -1 * hp["a"] ** 3 + hp["b"] ** 3 + hp["c"] - abs(hp["d"])

    hps = hp_module.HyperParameters()
    hps.Float("a", -1, 1)
    hps.Float("b", -1, 1)
    hps.Float("c", -1, 1)
    hps.Float("d", -1, 1)

    tuner = PolynomialTuner(
        oracle=keras_tuner.oracles.BayesianOptimizationTF(
            objective=keras_tuner.Objective("score", "max"),
            hyperparameters=hps,
            max_trials=20,
        ),
        directory=tmp_path,
    )
    benchmark(tuner.search)


@pytest.mark.benchmark(
    group="Float Optimization Benchmark of Numpy Einsum",
    min_rounds=10,
    max_time=6000,
)
def test_float_optimization_benchmark_np(tmp_path, benchmark):
    class PolynomialTuner(keras_tuner.engine.base_tuner.BaseTuner):
        def run_trial(self, trial):
            hp = trial.hyperparameters
            return -1 * hp["a"] ** 3 + hp["b"] ** 3 + hp["c"] - abs(hp["d"])

    hps = hp_module.HyperParameters()
    hps.Float("a", -1, 1)
    hps.Float("b", -1, 1)
    hps.Float("c", -1, 1)
    hps.Float("d", -1, 1)

    tuner = PolynomialTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"),
            hyperparameters=hps,
            max_trials=20,
        ),
        directory=tmp_path,
    )
    benchmark(tuner.search)


@pytest.mark.benchmark(
    group="Numerical Challenge Benchmark of TensorFlow - Maximize",
    min_rounds=10,
    max_time=6000,
)
def test_numeric_benchmark_tf(tmp_path, benchmark):
    class NumericTuner(keras_tuner.engine.base_tuner.BaseTuner):
        def run_trial(self, trial):
            hp = trial.hyperparameters
            return (
                -1 * hp["a"] ** 3
                + hp["b"] ** 3
                + hp["c"]
                - abs(hp["d"])
                + hp["e"] ** 3
                + hp["f"] ** 3
                + hp["g"]
                - abs(hp["h"])
            )

    hps = hp_module.HyperParameters()
    hps.Float("a", -1, 1)
    hps.Float("b", -1, 1)
    hps.Float("c", -1, 1)
    hps.Float("d", -1, 1)
    hps.Float("e", -1, 1)
    hps.Float("f", -1, 1)
    hps.Float("g", -1, 1)
    hps.Float("h", -1, 1)

    tuner = NumericTuner(
        oracle=keras_tuner.oracles.BayesianOptimizationTF(
            objective=keras_tuner.Objective("score", "max"),
            hyperparameters=hps,
            max_trials=30,
        ),
        directory=tmp_path,
    )
    benchmark(tuner.search)


@pytest.mark.benchmark(
    group="Numerical Challenge Benchmark of Numpy - Maximize",
    min_rounds=10,
    max_time=6000,
)
def test_numeric_benchmark_np(tmp_path, benchmark):
    class NumericTuner(keras_tuner.engine.base_tuner.BaseTuner):
        def run_trial(self, trial):
            hp = trial.hyperparameters
            return (
                -1 * hp["a"] ** 3
                + hp["b"] ** 3
                + hp["c"]
                - abs(hp["d"])
                + hp["e"] ** 3
                + hp["f"] ** 3
                + hp["g"]
                - abs(hp["h"])
            )

    hps = hp_module.HyperParameters()
    hps.Float("a", -1, 1)
    hps.Float("b", -1, 1)
    hps.Float("c", -1, 1)
    hps.Float("d", -1, 1)
    hps.Float("e", -1, 1)
    hps.Float("f", -1, 1)
    hps.Float("g", -1, 1)
    hps.Float("h", -1, 1)

    tuner = NumericTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"),
            hyperparameters=hps,
            max_trials=30,
        ),
        directory=tmp_path,
    )
    benchmark(tuner.search)
