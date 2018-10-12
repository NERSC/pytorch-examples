"""
Helper code for allocating slurm jobs and sending them work.
"""

from __future__ import print_function
import subprocess

def _run_command(cmd):
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception('Failed to run command: %s' % cmd)
    return result

def _run_command_async(cmd):
    return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class SlurmJob():
    """An object which manages a slurm allocation and submits tasks."""

    def __init__(self, node_type='haswell', n_nodes=1, qos='interactive', time=10):
        """Constructs the SlurmJob and requests an allocation"""
        self.node_type = node_type
        self.n_nodes = n_nodes
        # Configure the allocation
        cmd = 'salloc -C %s -N %i -q %s -t %s --no-shell' % (
            node_type, n_nodes, qos, time)
        # Request the allocation
        self.salloc_result = _run_command(cmd)
        # Extract the job ID
        self.jobid = self._extract_job_id(self.salloc_result)
        print(self.salloc_result.stderr.decode())

    def __del__(self):
        _run_command('scancel %i' % self.jobid)

    def _extract_job_id(self, salloc_result):
        out_str = salloc_result.stderr.decode()
        for line in out_str.split('\n'):
            if line.startswith('salloc: Granted job allocation'):
                return int(line.split()[4])
        # Something went wrong
        raise Exception('Failed to parse JOB ID from output: %s' % out_str)

    def submit_task(self, task_cmd, n_nodes=1):
        """Submit a task to run in the job allocation with srun"""
        cmd = 'srun --jobid %i -N %i %s' % (self.jobid, n_nodes, task_cmd)
        return _run_command_async(cmd)
