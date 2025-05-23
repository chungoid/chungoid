import sys
import os

# Add the 'src' directory to the Python path
# This allows pytest to find modules in the 'chungoid' package
# Assumes conftest.py is in the 'tests' directory, and 'src' is a sibling.
added_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, added_path)

# You can also add any project-wide fixtures here if needed in the future. 