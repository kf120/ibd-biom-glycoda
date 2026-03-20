# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import sys
from pathlib import Path

# Get the project root (parent of notebooks/*)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to sys.path so glaicoda can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Notebook config loaded. Project root added: {PROJECT_ROOT}")