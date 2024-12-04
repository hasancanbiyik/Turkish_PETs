QUICK START INSTRUCTIONS:

0. Prepare an experiment folder consisting of files called "train_X.csv", "val_X.csv", and "test_X_LANG.csv" (if only testing on one lang, keep LANG the same, e.g. "test_0_turkish.csv", "test_1_turkish.csv", etc.)

1. Make sure `src/launch.py` is correctly configured. (If you need to do more custom changes, modify `src/train.py`.

2. From a terminal, `cd` into the EXPERIMENT_RUNNER folder. Activate a virtual environment (e.g. using `python3 -m venv .venv; source .venv/bin/activate`).

3. If it's the first time using this EXPERIMENT_RUNNER folder, you need to:
    1. Make sure dependencies are installed once in the virtual env (in local/run.sh, uncomment `python3 -m pip install --no-cache-dir -r requirements.txt`)
    2. Enable permissions to execute the script (`chmod +x local/run.sh` should work)

4. Then, launch training using local/run.sh, e.g.:
    1. Regular command line: 
        `sh local/run.sh`
    2. Using SLURM (for long-term jobs)
        `srun --export=ALL --partition=gpu-week-long local/run.sh & disown`
        Also useful for SLURM is `sinfo` (displays status of compute nodes)
        
WHAT ARE THE DIFFERENT FILES DOING??
When you run local/run.sh, you install dependencies from requirements.txt and then launch src/launch.py, which creates a list of the different trials to run (src/manifest.py) and calls src/train.py on each of them, which does training/validation/testing and outputs metrics to your experiment directory.