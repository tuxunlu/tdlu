# config.py

import yaml

def load_config(config_path="config.yaml"):
    """
    Loads and returns the configuration dictionary from a YAML file.
    
    Parameters:
        config_path (str): Path to the config file. Default is "config.yaml".
    
    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
