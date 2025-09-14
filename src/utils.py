import yaml

# A helper function for deep merging dictionaries
def merge_configs(base, override):
    """Recursively merges override dict into base dict."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base:
            base[key] = merge_configs(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path):
    """Loads the base config and merges the experiment-specific config on top."""
    # 1. Load the base configuration that all experiments share
    with open("configs/base_config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    # 2. Load the specific experiment's configuration
    with open(config_path, 'r') as f:
        override_config = yaml.safe_load(f)

    # 3. Merge them, with the specific config overriding the base
    #    The result is your final, complete configuration dictionary.
    return merge_configs(base_config, override_config)