import os
from typing import List, Dict, Any, Tuple
from shutil import copytree
from pathlib import Path
import json
from subprocess import Popen, TimeoutExpired, PIPE, run
import os
import re
import shutil

from .datatypes import Task
CONDA_BIN = '/home/user/conda/bin/conda'


TIMEOUT = 30

def get_indent(code):
    line = code.split('\n')[0]
    return len(line) - len(line.strip())

def run_wrapper(cmd, cwd):
    my_env = os.environ.copy()
    my_env['PATH'] = f"{cwd}:" + my_env['PATH']
    my_env['PYTHONPATH'] = f"{cwd}"
    res = run([cmd.replace('\n', ' ')], shell=True, capture_output=True, check=False, env=my_env, timeout=TIMEOUT)
    return res.stdout.decode("utf-8") + res.stderr.decode("utf-8")


def run_tests(bin: os.PathLike, repo: os.PathLike) -> Dict[str, int]:
    """
    Execute all tests in the given path using pytest from bin
    """
    try:
        cmd = run_wrapper(f"cd {str(repo)} && conda run -p {str(bin)} pytest tests --color=no -p no:cacheprovider", cwd=str(repo))
    except TimeoutExpired:
        print('TIMEOUT CAUGHT')
        return {'passed': 0, 'failed': 0,  'output': 'TIMEOUT'}
    passed = re.findall(r" \d+ passed", cmd)
    if passed: 
        passed = int(passed[0][1:-7])
    else:
        passed = 0
    failed = re.findall(r" \d+ failed", cmd)
    if failed: 
        failed = int(failed[0][1:-7])
    else:
        failed = 0
    if cmd.find("short test summary info") != -1:
        out = '\n'.join(cmd.split('\n')[-50:]) # cmd[cmd.find("short test summary info"):]
    else:
        out = '\n'.join(cmd.split('\n')[:])
    return {'passed': passed, 'failed': failed, 'output': out}
            
def evaluate_override(
        root_path: os.PathLike, task: Task, generation: str, workdir: os.PathLike
) -> Dict[str, Any]:
    root_path  = Path(root_path)
    workdir = Path(workdir).absolute()
    if os.path.exists(workdir):
        try:
            shutil.rmtree(workdir)
        except FileNotFoundError as e:
            print(f"Caught file not found at rmtree {workdir}")
        workdir.mkdir(parents=True, exist_ok=True)
        
    copytree(root_path / task.repo, workdir, dirs_exist_ok=True, # we do not want to copy venv, it is very slow
        ignore=shutil.ignore_patterns(
            'venv_bench', '.github', '.git', '.pytest_cache', '*.egg-info', '__pycache__', 'testtemp'
        )
    )
    new_content = task.left_context + generation + task.right_context
    with open(workdir / task.path_from_root, 'w', encoding='utf-8') as f:
        f.write(new_content)

    metrics = run_tests(root_path / task.repo / "venv_bench", workdir)
    
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError as e:
        print(f"Caught file not found at rmtree {workdir}")
    except OSError as e:
        print(f"OSError {e} while rm {workdir}")
    return metrics

def evaluate_override_wrapped(
    root_path: os.PathLike, task: Task, generation: str, workdir: os.PathLike, task_n: int, gen_n: int, cache: dict
) -> Tuple[int, int, Dict[str, Any]]:
    cache_key = task.left_context + generation + task.right_context
    print('\r', task.repo, task.repo_n, cache_key in cache)
    if cache_key in cache:
        return (task_n, gen_n, cache[cache_key])
    else:
        res = evaluate_override(root_path, task, generation, workdir)
        cache[cache_key] = res
        return (task_n, gen_n, res)


def load_dataset(root_path: os.PathLike, meta_file: str = 'dataset.json', limit: int = 10_000) -> List[Task]:
    with open(Path(root_path) / meta_file, 'r') as f:
        dataset = [Task(**t) for t in json.load(f)][:limit]
    return dataset 