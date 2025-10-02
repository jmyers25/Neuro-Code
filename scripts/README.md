# Scripts

Shell helpers and orchestration tools intended to be human-run (not imported).

- **run_fit_rl.sh**  
  Example wrapper to launch training jobs (adjust paths/env as needed).

- **pred_rl_task_list.sh**  
  Task list for batch predictions (can be fed to a scheduler or GNU parallel).

- **run_parallel.py**  
  Parallel driver for simulations. Runs `get_predicted_df` across parameter ranges and repetitions
  using all available CPU cores (hyperthreading). Supports incremental or aggregated outputs,
  and records seeds/metadata for reproducibility.
