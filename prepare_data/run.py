import pandas as pd
from git import Repo
import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

os.environ['GIT_PYTHON_TRACE'] = 'full'

def clone_repos(target_dir):
    df = pd.read_csv('./repositories.csv')
    os.makedirs(target_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), 'Cloning repositories', total=df.shape[0]):
        if not os.path.exists(os.path.join(target_dir, row["name"])):
            repo = Repo.clone_from(row['url'], os.path.join(target_dir, row["name"]))
            repo.head.reset(row["commit"], index=True, working_tree=True)
        else:
            raise ValueError(f"Directory {os.path.join(target_dir, row['name'])} not empty")
    shutil.copy("./dataset.json", os.path.join(target_dir, 'dataset.json'))


def run(cmd):
    return subprocess.run(
        [cmd.replace('\n', ' ')], 
        shell=True, capture_output=True, check=False
    ).stdout.decode("utf-8")


# def setup(repo):
#     base_path = str(repo.resolve())
#     print(base_path)
#     if 'tests' not in os.listdir(base_path):
#         out = run(f"rm -rf {base_path}")
#         return None
        
#     run(f"rm -rf {base_path}/venv_bench")
#     run(f"rm -rf {base_path}/build")
#     d = run(f"./base_env/bin/python3 -m venv --copies {base_path}/venv_bench")

    
#     if os.path.exists(f"{base_path}/poetry.lock"):
#         out = run(f"cd {base_path} && ./base_env/bin/poetry export -o reqs_p.txt --without-hashes")
#         out = run(f"cd {base_path} && ./base_env/bin/poetry export --with dev -o reqs_p.txt --without-hashes")
#         out = run(f"cd {base_path} && ./base_env/bin/poetry export --with test -o reqs_p.txt --without-hashes")

#     for req_filename in ["reqs_p.txt", "requirements.txt", "linux_requirements.txt",
#          "requirements-ci.txt","requirements_ci.txt", "dev-requirements.txt",
#          'requirements_dev.txt', "requirements-dev.txt"]:
#         if os.path.exists(f"{base_path}/{req_filename}"):
#             out = run(f"{repo}/venv_bench/bin/python -m pip install -r {base_path}/{req_filename}")
#     out = run(f"{repo}/venv_bench/bin/python -m pip install -e {base_path}")
#     for toml_option in ["[test]", "[dev]", "[all]"]:
#         out = run(f"{repo}/venv_bench/bin/python -m pip install -e {base_path}.{toml_option}")
#     out = run(f"{repo}/venv_bench/bin/python -m pip install pytest")
#     return f"{repo}/venv_bench/bin/python"


def setup(repo):
    """
    builds conda environments in repo, that will be used to run tests
    """
    base_path = str(repo.resolve())
    run(f"rm -rf {base_path}/venv_bench")
    d = run(f"conda create -p {base_path}/venv_bench -y python=3.11 poetry")  
    if os.path.exists(f"{base_path}/poetry.lock"):
        out = run(f"cd {base_path} && conda run -p {base_path}/venv_bench poetry export -o reqs_p.txt --without-hashes")
        out = run(f"cd {base_path} && conda run -p {base_path}/venv_bench poetry export --with dev -o reqs_p.txt --without-hashes")
        out = run(f"cd {base_path} && conda run -p {base_path}/venv_bench poetry export --with test -o reqs_p.txt --without-hashes")
    for req_filename in ["reqs_p.txt", "requirements.txt", "linux_requirements.txt",
         "requirements-ci.txt","requirements_ci.txt", "dev-requirements.txt",
         'requirements_dev.txt', "requirements-dev.txt"]:
        if os.path.exists(f"{base_path}/{req_filename}"):
            out = run(f"conda run -p {base_path}/venv_bench python -m pip install -r {base_path}/{req_filename}")
    out = run(f"conda run -p {base_path}/venv_bench python -m pip install -e {base_path}")
    for toml_option in ["[test]", "[dev]", "[all]"]:
        out = run(f"conda run -p {base_path}/venv_bench python -m pip install {base_path}.{toml_option}")
    out = run(f"conda run -p {base_path}/venv_bench pip install pytest")
    return base_path


def build_envs(source_dir):
    repos_parent = Path(source_dir)
    for path in tqdm([t for t in repos_parent.iterdir() if os.path.isdir(t)], desc='building_envs'):
        setup(path)
    # run('rm -rf ./base_env')


if __name__ == '__main__':
    dataset_dir = '../data/realcode_v1'
    print(f'Clearing {dataset_dir}')
    run(f'rm -rf {dataset_dir}')
    print('Started collecting dataset')
    clone_repos(dataset_dir)
    print('All repos are cloned')
    build_envs(dataset_dir)
    print('Done')
