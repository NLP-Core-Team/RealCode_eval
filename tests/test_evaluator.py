import pytest
from pathlib import Path
import os
import json
import joblib

import lm_eval.utils
import lm_eval.evaluator

@pytest.fixture
def dataset_path():
    return str(Path('./data/realcode_v3').resolve())

def get_indent(code):
    line = code.split('\n')[0]
    return len(line) - len(line.strip())


@pytest.mark.parametrize('dataset_file', ['realcode_v3_SG.json', 'realcode_v3_FG.json'])
def test_perfect_preds(dataset_path, dataset_file, workdir='./workdir'):
    print("Testing where Pass@1 should be 1")
    root = Path(dataset_path)
    print(f"Dataset is at ", root, dataset_file)
    NJOBS = 8

    dataset = lm_eval.utils.load_dataset(root, dataset_file, limit=10_000)
    empty_ans = [[t.gt] for t in dataset]
    evaluator = lm_eval.evaluator.Evaluator(
        root,
        num_samples=1,
        pass_k_list=[1],
        njobs=NJOBS,
        working_dir=workdir
    )
    metrics = evaluator.evaluate(dataset, empty_ans)
    wrong = []
    for metric in metrics['detailed']:
        if  metric['Pass@1'] < 1 - 1e-3:
            wrong.append(metric)
            print(metric['Pass@1'], metric['repo'], metric['repo_n'], metric['path_from_root'], metric['evaluations'][0])
    with open('test_perfect_preds_fails.json', 'w') as f:
        json.dump(wrong, f)
    for x in wrong:
        print(x['repo'], x['path_from_root'], x['repo_n'])
    assert len(wrong) == 0


@pytest.mark.parametrize('dataset_file', ['realcode_v3_SG.json', 'realcode_v3_FG.json'])
def test_incorrect_answers(dataset_path, dataset_file):
    print("Testing where Pass@1 should be 0")
    root = Path(dataset_path)
    print(f"Dataset is at ", root, dataset_file)
    NJOBS = 8

    dataset = lm_eval.utils.load_dataset(root, dataset_file, limit=10_000)
    empty_ans = [[" "*get_indent(t.gt) + 'pass\n'] for t in dataset]
    evaluator = lm_eval.evaluator.Evaluator(
        root,
        1,
        [1],
        njobs=NJOBS,
        working_dir='./workdir'
    )
    metrics = evaluator.evaluate(dataset, empty_ans)
    wrong = []
    for metric in metrics['detailed']:
        if metric['Pass@1'] > 1e-3:
            wrong.append(metric)
            print(metric['Pass@1'], metric['repo'], metric['repo_n'], metric['path_from_root'], metric['evaluations'][0])
    print('\n' * 10)
    with open('test_incorrest_answers_fails.json', 'w') as f:
        json.dump(wrong, f)
    for x in wrong:
        print(x['repo'], x['path_from_root'], x['repo_n'])
    assert len(wrong) == 0



@pytest.mark.parametrize('dataset_file', ['realcode_v3_SG.json', 'realcode_v3_FG.json'])
def test_perfect_preds_parallel(dataset_path, dataset_file):
    """
    Like test_perfect_preds but with parallel evaluation
    """
    joblib.Parallel(n_jobs=8)(joblib.delayed(test_perfect_preds)(dataset_path, dataset_file, workdir=str(i)) for i in range(2))
