
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../benchmark/'))
# see: https://stackoverflow.com/a/7506029
import json

from common.utilities import JSON
from os import path
from pathlib import Path

def load_config():
    current_path = Path(__file__)
    config_path = path.join(
        current_path.parent.parent.absolute(), 'backend/config/')
    config = JSON.load_file(path.join(config_path, 'python.json'), 'r')
    return config
