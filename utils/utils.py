import os

def get_run_number(base_path="runs", prefix="mg_experiment_"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # List existing experiment directories that start with the prefix
    existing = [d for d in os.listdir(base_path) if d.startswith(prefix)]
    run_numbers = []
    for d in existing:
        # Remove the prefix and split by underscore
        remainder = d[len(prefix):]
        parts = remainder.split('_')
        if parts and parts[0].isdigit():
            run_numbers.append(int(parts[0]))
    if run_numbers:
        return max(run_numbers) + 1
    else:
        return 0
