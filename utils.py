import os

def get_run_number(base_path="runs", prefix="mg_experiment_"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # List existing experiment directories
    existing = [d for d in os.listdir(base_path) if d.startswith(prefix)]
    run_numbers = []
    for d in existing:
        try:
            run_numbers.append(int(d.replace(prefix, "")))
        except ValueError:
            continue
    if run_numbers:
        return max(run_numbers) + 1
    else:
        return 0