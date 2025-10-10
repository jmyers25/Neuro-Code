"""
run_parallel.py

Driver script to run get_predicted_df simulations in parallel across CPU cores.

- Uses ProcessPoolExecutor for true parallelism (avoids Python GIL).
- Sweeps across parameter ranges (Cartesian product).
- Supports multiple independent repetitions per parameter set.
- Each run gets its own RNG seed for reproducibility.
- Results can be written incrementally (per-task CSVs) or aggregated in-memory.

Recommended usage:
    python -m scripts.run_parallel
"""

import os
import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, Tuple, List

from neuro_code.helpers.get_predicted_df import get_predicted_df


def make_param_grid(param_ranges: Dict[str, Iterable]) -> List[Dict[str, float]]:
    """
    Expand a dictionary of parameter ranges into a list of all possible
    parameter combinations (Cartesian product).

    Args:
        param_ranges (Dict[str, Iterable]): Dictionary mapping parameter names
                                            to lists/ranges of values.

    Returns:
        List[Dict[str, float]]: Each element is a parameter dictionary with one
                                specific combination of parameter values.
    """
    keys = list(param_ranges.keys())
    combos = itertools.product(*(param_ranges[k] for k in keys))
    return [dict(zip(keys, vals)) for vals in combos]


def _run_single(
    df_csv_path: str,
    pars_dict: Dict[str, float],
    run_id: int,
    base_seed: int
) -> Tuple[int, Dict[str, float], pd.DataFrame]:
    """
    Worker function executed in parallel.

    Loads trial data, runs one simulation with the given parameter set,
    and annotates the output DataFrame with metadata.

    Args:
        df_csv_path (str): Path to trial CSV file.
        pars_dict (Dict[str, float]): Parameter values for this run.
        run_id (int): Unique run identifier.
        base_seed (int): Base seed to generate reproducible RNG stream.

    Returns:
        Tuple[int, Dict[str, float], pd.DataFrame]:
            (run_id, parameter dictionary, simulated DataFrame).
    """
    seed = base_seed + run_id
    rng = np.random.default_rng(seed)

    data = pd.read_csv(df_csv_path)
    sim_df = get_predicted_df(data, pars_dict, rng=rng)

    # Annotate with parameters and run metadata
    meta_cols = {f"param_{k}": v for k, v in pars_dict.items()}
    sim_df = sim_df.assign(run_id=run_id, seed=seed, **meta_cols)
    return run_id, pars_dict, sim_df


def run_hyperthreaded(
    df_csv_path: str,
    param_ranges: Dict[str, Iterable],
    n_reps: int = 1,
    max_workers: int = None,
    base_seed: int = 17,
    out_csv_path: str = "all_sims.csv",
    write_incremental: bool = False,
    out_dir: str = "sim_outputs"
) -> pd.DataFrame:
    """
    Run get_predicted_df across many parameter combinations in parallel.

    Args:
        df_csv_path (str): Path to trial CSV file.
        param_ranges (Dict[str, Iterable]): Dict mapping parameter names to lists of values.
        n_reps (int, optional): Number of repetitions per parameter set. Default = 1.
        max_workers (int, optional): Number of worker processes. Default = os.cpu_count().
        base_seed (int, optional): Base RNG seed for reproducibility. Default = 17.
        out_csv_path (str, optional): Path for aggregated output CSV. Default = "all_sims.csv".
        write_incremental (bool, optional): If True, writes each run to its own CSV
                                            in out_dir, then stitches them later.
        out_dir (str, optional): Directory for incremental CSV outputs. Default = "sim_outputs".

    Returns:
        pd.DataFrame: Aggregated DataFrame of all simulation runs.
    """
    os.makedirs(out_dir, exist_ok=True)
    param_grid = make_param_grid(param_ranges)

    # Build list of tasks
    tasks = []
    task_id = 0
    for rep in range(n_reps):
        for pars in param_grid:
            tasks.append((task_id, pars))
            task_id += 1

    if max_workers is None:
        max_workers = os.cpu_count() or 2

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_run_single, df_csv_path, pars, tid, base_seed)
            for tid, pars in tasks
        ]

        for fut in as_completed(futures):
            run_id, pars, df = fut.result()
            if write_incremental:
                # Write per-task file
                tag = "_".join(f"{k}-{pars[k]}" for k in sorted(pars))
                path = os.path.join(out_dir, f"sim_run-{run_id}_{tag}.csv")
                df.to_csv(path, index=False)
            else:
                results.append(df)

    if write_incremental:
        # Merge all per-task CSVs
        all_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".csv")]
        big = pd.concat((pd.read_csv(p) for p in all_files), ignore_index=True)
        big.to_csv(out_csv_path, index=False)
        return big
    else:
        big = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if not big.empty:
            big.to_csv(out_csv_path, index=False)
        return big


if __name__ == "__main__":
    # Example usage with placeholder parameter ranges and data path.
    param_ranges = {
        "beta":    [0.01, 0.05, 0.1],
        "lossave": [0.5, 1.0],
        "alpha":   [0.05, 0.1],  # for symmetric alpha
        "exp":     [1.0],        # for symmetric exp
    }

    TRIAL_DATA_PATH = "path/to/trial_data.csv"

    out = run_hyperthreaded(
        df_csv_path=TRIAL_DATA_PATH,
        param_ranges=param_ranges,
        n_reps=5,
        max_workers=None,         # uses all available logical cores
        base_seed=1337,
        out_csv_path="all_sims.csv",
        write_incremental=False,  # set True if you want per-task files in sim_outputs/
        out_dir="sim_outputs"
    )
    print("Finished. Rows:", len(out))
