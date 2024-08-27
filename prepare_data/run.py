import pandas as pd
from git import Repo
import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

os.environ['GIT_PYTHON_TRACE'] = 'full'


def run(cmd, check=False):
    return subprocess.run(
        [cmd.replace('\n', ' ')], 
        shell=True, capture_output=True, check=check,
    ).stdout.decode("utf-8")

def delete_and_report(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False)
        if os.path.exists(path):
            raise ValueError(f"Unable to delete {path}, please delete it manually")

def setup(repo):
    """
    builds conda environments in repo, that will be used to run tests
    """
    base_path = str(repo.resolve().absolute())
    delete_and_report(f"{base_path}/venv_bench")
    delete_and_report(f"{base_path}/build")
    delete_and_report(f"{base_path}/*.egg-info")
    try:
        d = run(f"conda create -p {base_path}/venv_bench --copy -y python=3.11 poetry", check=True)      
    except subprocess.CalledProcessError as e:
        print(repo, 'create')
        print(e.stdout)
        print(e.stderr)
        raise e
    if os.path.exists(f"{base_path}/poetry.lock"):
        run(f"rm {base_path}/reqs_p.txt")
        out = run(f"cd {base_path} && conda run -p {base_path}/venv_bench poetry export -o reqs_p.txt --without-hashes")
        out = run(f"cd {base_path} && conda run -p {base_path}/venv_bench poetry export --with dev -o reqs_p.txt --without-hashes")
        out = run(f"cd {base_path} && conda run -p {base_path}/venv_bench poetry export --with test -o reqs_p.txt --without-hashes")
    
    for req_filename in ["reqs_p.txt", "requirements.txt", "linux_requirements.txt",
        "requirements-ci.txt","requirements_ci.txt", "dev-requirements.txt",
        'requirements_dev.txt', "requirements-dev.txt"]:
        if os.path.exists(f"{base_path}/{req_filename}"):
            out = run(f"conda run -p {base_path}/venv_bench python -m pip install -r {base_path}/{req_filename}", check=True)
    skip_install = False
    try: 
        if not skip_install and (os.path.exists(f"{base_path}/setup.py") or os.path.exists(f"{base_path}/pyproject.toml")):
            out = run(f"conda run -p {base_path}/venv_bench python -m pip install {base_path}", check=True)        
    except subprocess.CalledProcessError as e:
        print('='*40)
        print(repo, 'pip install warn')
        print('='*40)
    for toml_option in ["[test]", "[dev]", "[all]"]: 
        out = run(f"conda run -p {base_path}/venv_bench python -m pip install {base_path}.{toml_option}")
    out = run(f"conda run -p {base_path}/venv_bench pip install pytest")
    if not os.path.exists(f"{repo}/venv_bench/bin/python"):
        raise ValueError(f"{repo}/venv_bench/bin/python not found")
    print(repo, "done")
    return base_path


def build_envs(source_dir):
    repos_parent = Path(source_dir)
    Parallel(n_jobs=8)(
        delayed(setup)(path)
        for path in tqdm([t for t in repos_parent.iterdir() if os.path.isdir(t)], desc='building_envs')
    )



if __name__ == '__main__':
    dataset_dir = '../data/realcode_v3'
    build_envs(dataset_dir)
    print('Done')
