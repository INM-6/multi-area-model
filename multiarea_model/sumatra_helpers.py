from config import data_path
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
import os
import glob
import json
import numpy as np


def register_record(label, reason=None, tag=None):
    """
    Register a simulation to the sumatra project.
    Loads the sumatra project in the current repository

    Parameters
    ----------
    label : str
         Simulation label
    reason : str, optional
         Reason for the simulation run stored in the sumatra database
    tag : str, optional
         Tag for the simulation run stored in the sumatra database
    """

    project = load_project()

    para_fn = os.path.join(data_path,
                           label,
                           '_'.join(('custom_params',
                                     label)))
    parameters = build_parameters(para_fn)

    record = project.new_record(parameters=parameters,
                                main_file='nest_simulation.py',
                                reason=reason,
                                label=label)
    record.duration = 0.  # Add 0 for now and update later
    project.add_record(record)

    project.save()

    if tag is not None:
        project.add_tag(label, tag)


def register_runtime(label):
    """
    Register the duration of simulation run in the sumatra database.
    Loads the runtime automatically from the logfiles in the simulation
    directory.

    Parameters
    ----------
    label : str
         Simulation label
    """
    fp = os.path.join(data_path,
                      label,
                      'recordings',
                      'runtime_*')
    files = glob.glob(fp)

    for i, fn in enumerate(files):
        with open(fn, 'r') as f:
            d = json.load(f)
        if i == 0:
            data = {key: [value] for key, value in d.items()}
        else:
            for key, value in d.items():
                data[key].append(value)
    for key, value in data.items():
        data[key] = np.mean(value)

    project = load_project()
    record = project.get_record(label)
    record.duration = sum(data.values())
    project.add_record(record)
    project.save()
