
import hydra
import torch
import numpy as np
import random
import json
import os

from lm_eval.generators import InfillGenerator, LMGenerator
from lm_eval.evaluator import Evaluator
from lm_eval.context_parser import TrivialContextParser
from lm_eval.utils import load_dataset

from omegaconf import DictConfig, OmegaConf


import logging
logger = logging.getLogger("RealCode")
logger.setLevel(logging.DEBUG)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_all(cfg.seed)
    print(cfg)

    dataset = load_dataset(cfg.dataset_root, cfg.dataset_meta_file, cfg.limit)
    logger.info(f"loaded {cfg.dataset_root} {cfg.dataset_meta_file}")

    if 'context_parser' in cfg:
        parser = hydra.utils.instantiate(cfg.context_parser)
    else:
        parser = TrivialContextParser()

    dtype_map = {'fp16': torch.float16, 'fp32': torch.float, 'bf16': torch.bfloat16}
    if cfg.generator_mode == 'infill':
        generator = InfillGenerator(
            add_extra_spaces_to_begin=0,
            model_path=cfg.model_path,
            dtype=dtype_map[cfg.dtype],
            num_samples=cfg.num_samples,
            prefix_tokens=cfg.prefix_tokens,
            middle_tokens=cfg.middle_tokens,
            suffix_tokens=cfg.suffix_tokens,
            max_context_length=cfg.max_context_length,
            generation_params=dict(cfg.generation_params),
            eos_sequences=cfg.eos_sequences,
            model_kwargs=cfg.model_kwargs if 'model_kwargs' in cfg else {},
            context_parser=parser,
            left_context_ratio=cfg.left_context_ratio,
            add_extra_spaces_to_generation=cfg.tokenizer_fix
        )
    elif cfg.generator_mode == 'lm':
        generator = LMGenerator(
            model_path=cfg.model_path,
            dtype=dtype_map[cfg.dtype],
            num_samples=cfg.num_samples,
            lm_prefix_tokens=cfg.lm_prefix_tokens if 'lm_prefix_tokens' in cfg else [],
            lm_suffix_tokens=cfg.lm_suffix_tokens if 'lm_suffix_tokens' in cfg else [],
            max_context_length=cfg.max_context_length,
            generation_params=dict(cfg.generation_params),
            eos_sequences=cfg.eos_sequences,
            model_kwargs=cfg.model_kwargs if 'model_kwargs' in cfg else {},
            context_parser=parser,
            add_extra_spaces_to_generation=cfg.tokenizer_fix
        )
    else:
        raise ValueError(f"generator_mode can be either 'lm' or 'infill', found {cfg.generator_mode}")
    evaluator = Evaluator(
        dataset_root=cfg.dataset_root,
        num_samples=cfg.num_samples,
        pass_k_list=cfg.pass_k_list,
        njobs=cfg.njobs,
        working_dir=cfg.working_dir,
    )

    logger.info(f"Starting generation")
    if cfg.use_cached_generations and os.path.exists(cfg.generations_save_path):
        with open(cfg.generations_save_path, "r") as f:
            generations = json.load(f)
    else:
        generations = generator.generate(dataset)
        with open(cfg.generations_save_path, "w") as f:
            json.dump(generations, f)

    del generator.model # clean up cuda, we don't need it

    logger.info(f"Starting evaluation")
    metrics = evaluator.evaluate(dataset, generations)
    if generator.count_inferenced_tokens and generator.count_possible_tokens:
        # a bit ugly, otherwise breaks when loading generations from disk
        metrics['total']['avg_used_tokens'] = (
            sum([t[0] for t in generator.count_inferenced_tokens]) 
            / len(generator.count_inferenced_tokens) 
        )
        metrics['total']['avg_possible_tokens'] = (
            sum([t for t in generator.count_possible_tokens]) 
            / len(generator.count_possible_tokens) 
        )
    logger.info(metrics['total'])
    if cfg.metrics_save_path:
        with open(cfg.metrics_save_path, "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()

