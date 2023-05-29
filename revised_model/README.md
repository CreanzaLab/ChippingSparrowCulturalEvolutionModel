## A revised, faster-running model

Run the model with Python in a directory that contains an `out_dir` directory:

```
$ python3 run_bird_model.py "neutral"|"conformity"|"directional" dispersal_rate
    error_rate_percentage [-d matrix_size] [-s simulation_number] [-t total_iterations]
    [--high_syll "constant"|"adaptive"]
```

After these have been run with the desired parameters, analyze results as follows:

```
$ python3 fisher_testing.py matrix_size
```
