# Adaptive Long Term Memory for Multitask Bandits
Source code for the experiments accompanying the paper entitled "Adaptive Long Term Memory for Multitask Bandits"

## Requirements
* Python >= 3.7
* Numpy >= 1.19

## Instructions
Run main.py with the config_name set to the name of the desired configuration file, which is located in /config/ .

The configuration files define experimental controls, and have the following form:

```
{
  "description": "",
  "algorithms": [...], // a list of the algorithms to run 
  "global": {...}, // definitions of global parameters which apply to the experiment
  "local": {...}, // definitions of parameters which apply to individual algorithms
  "tuning_parameters": {...} // definitions of parameters for tuning/optimising algorithm parameters
}
```

For example 

```
{
  "description": "Config for multitask experiments",
  "algorithms": [
    "ADAPTLTM",
    "MARKOV_SPECIALISTS_EXP4",
    "SWARM",
    "FIXED_SHARE_EXP4",
    "EXP4"
  ],
  "global": {
    "n_tasks": 12,
    "n_actions": 16,
    "total_trials_global": 28800,
    "n_epochs": 10,
    "n_switches_per_epoch": 11,
    "super_pool_size": 80,
    "epoch_pool_size": 4,
    "epoch_block_pool_size": 12,
    "random_switches_within_epoch": true,
    "good_loss_upper_bound": 0.025,
    "bad_loss_upper_bound": 0.5,
    "n_experts": 1024,
    "epsilon": 0.1,
    "n_runs": 500
  },
  "local": {
    "ADAPTLTM": {
      "eta_optimal": false,
      "eta": 13.771428571428572,
      "alpha": 0.25,
      "rho": 11
    },
    "SWARM": {
      "eta_optimal": false,
      "eta": 15.142857142857142
    },
    "EXP4": {
      "eta_optimal": false,
      "eta": 11.028571428571428
    },
    "MARKOV_SPECIALISTS_EXP4": {
      "eta_optimal": false,
      "eta": 15.485714285714286,
      "alpha_optimal": true,
      "alpha": 0.0,
      "theta_optimal": true,
      "theta": 0.0
    },
    "FIXED_SHARE_EXP4": {
      "eta_optimal": false,
      "eta": 17.2,
      "alpha_optimal": true,
      "alpha": 0.0
    }
  },
  "tuning_parameters": {
    "n_iterations_per_training_step": 8,
    "n_tuning_steps": 36,
    "n_processes": 12,
    "eta_min": 10.0,
    "eta_max": 22.0,
    "seed": 888888
  }
}
```

The "n_processes" parameter in "tuning_parameters" also applies when running main.py itself, as this uses python's 
multiprocessing library.

Results are stored as a pickled dictionary object in /output/ . The keys of the dictionary are the algorithm names from the configuration file, and the values are numpy arrays of shape (n_runs, total_trials_global), containing losses from each run.
