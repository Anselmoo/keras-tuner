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

import time

import keras_tuner
from keras_tuner.engine import hyperparameters as hp_module


class NumericTuner(keras_tuner.engine.base_tuner.BaseTuner):
    def run_trial(self, trial):
        hp = trial.hyperparameters
        return (
            -1 * hp["a"] ** 3
            + hp["b"] ** 2 / hp["c"]
            - abs(hp["d"])
            + hp["e"] ** 3
            + hp["f"] ** 2 / hp["g"]
            - abs(hp["h"])
            + hp["i"] ** 3
            + hp["j"] ** 2 / hp["k"]
            - abs(hp["l"])
            + hp["m"] ** 3
            + hp["n"] ** 2
            + hp["o"]
            - abs(hp["p"])
            + hp["q"] ** 3
            + hp["r"] ** 2 / hp["s"]
            - abs(hp["t"])
        )


def hyperparameter_benchmark() -> hp_module.HyperParameters:
    hps = hp_module.HyperParameters()
    hps.Float("a", -10, 10)
    hps.Float("b", -10, 10)
    hps.Float("c", -10, 10)
    hps.Float("d", -10, 10)
    hps.Float("e", -10, 10)
    hps.Float("f", -10, 10)
    hps.Float("g", -10, 10)
    hps.Float("h", -10, 10)
    hps.Float("i", -10, 10)
    hps.Float("j", -10, 10)
    hps.Float("k", -10, 10)
    hps.Float("l", -10, 10)
    hps.Float("m", -10, 10)
    hps.Float("n", -10, 10)
    hps.Float("o", -10, 10)
    hps.Float("p", -10, 10)
    hps.Float("q", -10, 10)
    hps.Float("r", -10, 10)
    hps.Float("s", -10, 10)
    hps.Float("t", -10, 10)
    return hps


def benchmark_np(round: int = 3) -> None:

    start = time.time()

    for i in range(round):
        tuner_np = NumericTuner(
            oracle=keras_tuner.oracles.BayesianOptimization(
                objective=keras_tuner.Objective("score", "max"),
                hyperparameters=hp,
                max_trials=80,
            ),
            directory=f"./np_{i}",
        )
        tuner_np.search()
        print(
            "Current time for Numpy Oracle: ",
            time.time() - start,
            "with score: ",
            tuner_np.oracle.get_best_trials()[0].score,
        )
    print("Average time for Numpy Oracle: ", (time.time() - start) / round)
    print("Total time for Numpy Oracle: ", time.time() - start)
    print(tuner_np.oracle.get_best_trials()[0].score)


def benchmark_tf(round: int = 3) -> None:

    start = time.time()

    for i in range(round):
        tuner_tf = NumericTuner(
            oracle=keras_tuner.oracles.BayesianOptimization(
                objective=keras_tuner.Objective("score", "max"),
                hyperparameters=hp,
                max_trials=80,
            ),
            directory=f"./tf_{i}",
        )
        tuner_tf.search()
        print(
            "Current time for TensorFlow Oracle: ",
            time.time() - start,
            "with score: ",
            tuner_tf.oracle.get_best_trials()[0].score,
        )
    print("Average time for TensorFlow Oracle: ", (time.time() - start) / round)
    print("Total time for TensorFlow Oracle: ", time.time() - start)


if __name__ == "__main__":

    hp = hyperparameter_benchmark()

    print("Benchmarking Numpy Oracle\n\n")
    benchmark_np()
    print("Benchmarking TensorFlow Oracle\n\n")
    benchmark_tf()
