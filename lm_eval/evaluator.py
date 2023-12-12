import os
import typing as tp
import math
from collections import defaultdict
import json
import re
from statistics import mean
from dataclasses import asdict
from multiprocessing import Pool, Manager

from .utils import evaluate_override, evaluate_override_wrapped
from .datatypes import Task

import logging
logger = logging.getLogger("RealCode")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_num_lines_bin(t: Task):
    lines = t.gt.count('\n') - 1
    if 1 <= lines <= 2:
        return '1-2'
    elif 3 <= lines <= 5:
        return '3-5'
    elif 6 <= lines <= 10:
        return '6-10'
    elif lines > 10:
        return '10+'


METRIC_AGGREGATIONS = {
    'total': lambda t: 1, 
    'repo': lambda t: t.repo,
    'nlines_bin': get_num_lines_bin,
    # 'detailed': lambda t: t,
}

class PassK:
    def __init__(self, k: int, n: int):
        self.k = k
        self.n = n

    def __call__(self, correct: int):
        return (1 - (math.comb(self.n - correct, self.k) / math.comb(self.n, self.k)))
    
    def name(self):
        return f"Pass@{self.k}"


class Evaluator:
    def __init__(self, 
        dataset_root: os.PathLike,
        num_samples: int,
        pass_k_list: tp.List[int] = [1],
        njobs: int = 1,
        working_dir: tp.Optional[os.PathLike] = None,
        metric_aggregations: tp.Dict[str, tp.Callable[[Task], int]] = METRIC_AGGREGATIONS
    ):
        self.metrics = []
        for pass_k in  pass_k_list:
            if num_samples < pass_k:
                raise ValueError(f"num_samples {num_samples} must be greater than or equal to PassK={pass_k}")
            self.metrics.append(PassK(pass_k, num_samples))
        self.dataset_root = dataset_root
        self.num_samples = num_samples
        self.njobs = njobs
        self.working_dir = working_dir
        self.metric_aggregations = metric_aggregations
        
    def evaluate(self, 
        tasks: tp.List[Task],
        generations: tp.List[tp.List[str]],
    ) -> tp.Dict[tp.Literal["aggregated", "detailed"], tp.Any]:
        logger.info(f"Evaluating {len(tasks)} tasks with {self.num_samples} samples on {self.njobs} CPUs")
        # Run test evaluation
        if self.njobs == 1:
            results = [
                [evaluate_override( self.dataset_root, task, gen, os.path.join(self.working_dir) ) for gen in generations[i]]
                for i, task in enumerate(tasks)
            ]
        else:
            with Manager() as manager:
                cache = manager.dict()
                with manager.Pool(processes=self.njobs) as pool:
                    results = [[None for _2 in range(self.num_samples)] for _ in tasks]
                    async_result = pool.starmap_async(
                        evaluate_override_wrapped, [
                            ( self.dataset_root, task, gen, os.path.join(self.working_dir, f"{j}_{i}"), j, i, cache )
                                for j, task in enumerate(tasks) for i, gen in enumerate(generations[j])
                        ]
                    )
                    res = async_result.get()
                    for task_n, gen_n, result in res:
                        results[task_n][gen_n] = result
                        if task_n % 25 == 0 and gen_n == 0:
                            logger.debug(result['output'])

        # Calculate metrics per task
        all_metric_names = ['compilation_error_rate', 'exact_match'] + [t.name() for t in self.metrics]
        metrics = []
        agg_metrics = {level: {metric_name: defaultdict(list) for metric_name in all_metric_names} for level in self.metric_aggregations}
        for task, task_results, task_generations in zip(tasks, results, generations):
            if len(task_results) != self.num_samples:
                raise ValueError(f"Task {task} has {len(task_results)} samples, expected {self.num_samples}")
            correct = sum([int(t['passed'] == task.total_tests) for t in task_results])
            not_compiles = mean([int(t['passed'] + t['failed'] == 0) for t in task_results])
            exact_match = mean([int(re.sub(r'\W+', '', task.gt) == re.sub(r'\W+', '', gen)) for gen in task_generations])
            task_metrics = {'compilation_error_rate': not_compiles, 'exact_match': exact_match}
            for metric in self.metrics:
                task_metrics[metric.name()] = metric(correct)
            task_metrics['evaluations'] = [t['output'] for t in task_results]
            metrics.append(task_metrics)
            for level, level_func in self.metric_aggregations.items():
                for metric in all_metric_names:
                    agg_metrics[level][metric][level_func(task)].append(task_metrics[metric])
   
        for level in self.metric_aggregations:
            for metric_name in all_metric_names:
                means = {val: mean(agg_metrics[level][metric_name][val]) for val in agg_metrics[level][metric_name]}
                agg_metrics[level][metric_name] = means

        # Save metics
        metrics = agg_metrics | {
            "detailed": [asdict(task) | task_metric for task, task_metric in zip(tasks, metrics)]
        }
        return metrics
