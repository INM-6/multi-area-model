import json
import nest
import os
import sys

from config import data_path
from multiarea_model import MultiAreaModel

label = sys.argv[1]
network_label = sys.argv[2]
fn = os.path.join(data_path,
                  label,
                  '_'.join(('custom_params',
                            label,
                           str(nest.Rank()))))
with open(fn, 'r') as f:
    custom_params = json.load(f)

os.remove(fn)

M = MultiAreaModel(network_label,
                   simulation=True,
                   sim_spec=custom_params['sim_params'])
M.simulation.simulate()
