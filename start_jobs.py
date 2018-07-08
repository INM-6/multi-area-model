import json
import os
import shutil

from config import base_path, data_path
from multiarea_model.default_params import nested_update, sim_params
try:
    from multiarea_model.sumatra_helpers import register_record
    sumatra_found = True
except ImportError:
    sumatra_found = False


def start_job(label, submit_cmd, jobscript_template, sumatra=False, reason=None, tag=None):
    """
    Start job on a compute cluster.

    Parameters
    ----------

    label : str
        Simulation label identifying the simulation to be run.
        The function loads all necessary files from the subfolder
        identified by the label.
    submit_cmd : str
        Submit command of the queueing system used.
    jobscript_template : formatted str
        Formatted string defining the template for the job script.
        Can include the following keyword arguments:
            sim_dir : str
                Directory of the simulation
            label : str
                Simulation label
            num_processes : int
                Total number of MPI processes, defined in sim_params
            local_num_threads : int
                Number of OpenMP threads per MPI process, defined in sim_params
            base_path : str
                Base path of the library defined in config.py
    """

    # Copy run_simulation script to simulation folder
    shutil.copy2(os.path.join(base_path, 'run_simulation.py'),
                 os.path.join(data_path, label))

    # Load simulation parameters
    fn = os.path.join(data_path,
                      label,
                      '_'.join(('custom_params',
                                label)))
    with open(fn, 'r') as f:
        custom_params = json.load(f)
    nested_update(sim_params, custom_params['sim_params'])

    # Copy custom param file for each MPI process
    for i in range(sim_params['num_processes']):
        shutil.copy(fn, '_'.join((fn, str(i))))
    # Collect relevant arguments for job script
    num_vp = sim_params['num_processes'] * sim_params[
        'local_num_threads']
    d = {'label': label,
         'network_label': custom_params['network_label'],
         'base_path': base_path,
         'sim_dir': os.path.join(data_path, label),
         'local_num_threads': sim_params['local_num_threads'],
         'num_processes': sim_params['num_processes'],
         'num_vp': num_vp}

    # Write job script
    job_script_fn = os.path.join(data_path,
                                 label,
                                 '_'.join(('job_script',
                                           '.'.join((label, 'sh')))))
    with open(job_script_fn, 'w') as f:
        f.write(jobscript_template.format(**d))

    # If chosen, register simulation to sumatra
    if sumatra:
        if sumatra_found:
            register_record(label, reason=reason, tag=tag)
        else:
            raise ImportWarning('Sumatra is not installed, so'
                                'cannot register simulation record.')

    # Submit job
    os.system('{submit_cmd} {job_script_fn}'.format(submit_cmd=submit_cmd,
                                                    job_script_fn=job_script_fn))
