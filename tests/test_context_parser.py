import pytest
from pathlib import Path
import os
import json
import numpy as np

import lm_eval.utils
import lm_eval.evaluator
from lm_eval.context_parser import SmartContextParser

@pytest.fixture()
def dataset_path():
    return './data/realcode'


def get_indent(code):
    line = code.split('\n')[0]
    return len(line) - len(line.strip())


def test_parsing_full_context(dataset_path):
    root = Path(dataset_path)
    

    dataset = lm_eval.utils.load_dataset(root, 'dataset.json', limit=10_000)
    context_parser = SmartContextParser(
        left_config=['imports', 'file', 'outer', 'inner'],
        right_config=['outer', 'file']
    )
    for i,task in enumerate(dataset):
        print(i)
        left_parsed, right_parsed = context_parser.get_left_and_right_context(task)
        print(right_parsed, "AAAA")
        set_left_parsed = set(left_parsed.split('\n'))
        set_right_parsed = set(right_parsed.split('\n'))

        set_left_gt = set(task.left_context.split('\n'))
        set_right_gt = set(task.right_context.split('\n'))

        assert not list(np.setdiff1d(set_left_gt, set_left_parsed))
        assert not list(np.setdiff1d(set_left_parsed, set_left_gt))
        assert not list(np.setdiff1d(set_right_gt, set_right_parsed))
        assert not list(np.setdiff1d(set_right_parsed, set_right_gt))


if __name__ == '__main__':
    test_parsing_full_context('./data/realcode')
    