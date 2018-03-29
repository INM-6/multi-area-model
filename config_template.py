# Absolute path of repository
base_path = None
# Place to store simulations
data_path = None
# Template for job scripts
jobscript_template = """
# Instruction for the queuing system

mpirun python {base_path}/run_simulation.py {label}"""

# Command to submit jobs on the local cluster
submit_cmd = None
