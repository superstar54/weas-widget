import os
from appdirs import user_config_dir

# Define the application name for storing configuration files
APP_NAME = "weas-widget"

# Get the appropriate configuration directory for the system
CONFIG_DIR = user_config_dir(APP_NAME)

# Ensure the directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

# Default server settings
DEFAULT_PORT = 8000
