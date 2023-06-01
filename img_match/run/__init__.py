import os
import sys
def _load_module(module_name):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dataset=os.path.join(root_dir,module_name)
    sys.path.insert(1, load_dataset)