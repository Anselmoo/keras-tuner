# Notes about the benchmarking process.


## General test `bayesian` vs `bayesian_tf`

```shell
collected 30 items

keras_tuner/tuners/bayesian_test.py::test_scipy_not_install_error PASSED                                                                                                                                                                                                  [  3%]
keras_tuner/tuners/bayesian_test.py::test_gpr_mse_is_small PASSED                                                                                                                                                                                                         [  6%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_oracle PASSED                                                                                                                                                                                                          [ 10%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_oracle_with_zero_y PASSED                                                                                                                                                                                              [ 13%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_dynamic_space PASSED                                                                                                                                                                                                   [ 16%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_save_reload PASSED                                                                                                                                                                                                     [ 20%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_optimization_tuner PASSED                                                                                                                                                                                              [ 23%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_optimization_tuner_set_alpha_beta PASSED                                                                                                                                                                               [ 26%]
keras_tuner/tuners/bayesian_test.py::test_save_before_result PASSED                                                                                                                                                                                                       [ 30%]
keras_tuner/tuners/bayesian_test.py::test_bayesian_oracle_maximize PASSED                                                                                                                                                                                                 [ 33%]
keras_tuner/tuners/bayesian_test.py::test_hyperparameters_added PASSED                                                                                                                                                                                                    [ 36%]
keras_tuner/tuners/bayesian_test.py::test_step_respected PASSED                                                                                                                                                                                                           [ 40%]
keras_tuner/tuners/bayesian_test.py::test_float_optimization PASSED                                                                                                                                                                                                       [ 43%]
keras_tuner/tuners/bayesian_test.py::test_distributed_optimization PASSED                                                                                                                                                                                                 [ 46%]
keras_tuner/tuners/bayesian_test.py::test_interleaved_distributed_optimization PASSED                                                                                                                                                                                     [ 50%]
keras_tuner/tuners/bayesian_test_tf.py::test_scipy_not_install_error PASSED                                                                                                                                                                                               [ 53%]
keras_tuner/tuners/bayesian_test_tf.py::test_gpr_mse_is_small PASSED                                                                                                                                                                                                      [ 56%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_oracle PASSED                                                                                                                                                                                                       [ 60%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_oracle_with_zero_y PASSED                                                                                                                                                                                           [ 63%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_dynamic_space PASSED                                                                                                                                                                                                [ 66%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_save_reload PASSED                                                                                                                                                                                                  [ 70%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_optimization_tuner PASSED                                                                                                                                                                                           [ 73%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_optimization_tuner_set_alpha_beta PASSED                                                                                                                                                                            [ 76%]
keras_tuner/tuners/bayesian_test_tf.py::test_save_before_result PASSED                                                                                                                                                                                                    [ 80%]
keras_tuner/tuners/bayesian_test_tf.py::test_bayesian_oracle_maximize PASSED                                                                                                                                                                                              [ 83%]
keras_tuner/tuners/bayesian_test_tf.py::test_hyperparameters_added PASSED                                                                                                                                                                                                 [ 86%]
keras_tuner/tuners/bayesian_test_tf.py::test_step_respected PASSED                                                                                                                                                                                                        [ 90%]
keras_tuner/tuners/bayesian_test_tf.py::test_float_optimization PASSED                                                                                                                                                                                                    [ 93%]
keras_tuner/tuners/bayesian_test_tf.py::test_distributed_optimization PASSED                                                                                                                                                                                              [ 96%]
keras_tuner/tuners/bayesian_test_tf.py::test_interleaved_distributed_optimization PASSED                                                                                                                                                                                  [100%]

======================================================================================================================== 30 passed in 483.24s (0:08:03) =========================================================================================================================
```

## Benchmark `bayesian` vs `bayesian_tf`

```shell
keras_tuner/tuners/benchmark_test.py::test_float_optimization_benchmark_tf PASSED                                                                                                                                                             [ 25%]
keras_tuner/tuners/benchmark_test.py::test_float_optimization_benchmark_np PASSED                                                                                                                                                             [ 50%]
keras_tuner/tuners/benchmark_test.py::test_numeric_benchmark_tf PASSED                                                                                                                                                                        [ 75%]
keras_tuner/tuners/benchmark_test.py::test_numeric_benchmark_np PASSED                                                                                                                                                                        [100%]


------------------------------------ benchmark 'Float Optimization Benchmark of Numpy Einsum': 1 tests -------------------------------------
Name (time in us)                             Min       Max      Mean   StdDev    Median     IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------
test_float_optimization_benchmark_np     144.4030  356.8090  149.7184  14.4148  146.5040  1.4000    45;116        6.6792     978           1
--------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------- benchmark 'Float Optimization Benchmark of TensorFlow Einsum': 1 tests ----------------------------------
Name (time in us)                             Min       Max      Mean   StdDev    Median      IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------
test_float_optimization_benchmark_tf     146.1030  292.8070  173.1797  43.5019  148.6040  26.8020     49;49        5.7743     234           1
---------------------------------------------------------------------------------------------------------------------------------------------

----------------------------- benchmark 'Numerical Challenge Benchmark of Numpy - Maximize': 1 tests ----------------------------
Name (time in us)                  Min       Max      Mean   StdDev    Median     IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------
test_numeric_benchmark_np     275.2050  465.1090  283.7981  22.1247  278.9555  2.3000     19;55        3.5236     444           1
---------------------------------------------------------------------------------------------------------------------------------

-------------------------- benchmark 'Numerical Challenge Benchmark of TensorFlow - Maximize': 1 tests ---------------------------
Name (time in us)                  Min       Max      Mean   StdDev    Median      IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------
test_numeric_benchmark_tf     275.3070  527.2120  319.3230  78.0048  278.9070  28.5268     17;22        3.1316     103           1
----------------------------------------------------------------------------------------------------------------------------------
```