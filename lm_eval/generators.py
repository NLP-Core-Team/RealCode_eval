import os
import typing as tp
import json
from pathlib import Path
from dataclasses import asdict, fields
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
from tqdm import tqdm

from .datatypes import Task
from .context_parser import BaseParser, TrivialContextParser

import logging
logger = logging.getLogger("RealCode")



class InfillGenerator:
    def __init__(self, 
        model_path: str,
        num_samples: int,
        prefix_tokens: tp.Union[str, tp.List[int]] = [],
        middle_tokens: tp.Union[str, tp.List[int]] = [],
        suffix_tokens: tp.Union[str, tp.List[int]] = [],
        max_context_length: int = None,
        left_context_ratio: int = 1,
        dtype = torch.bfloat16,
        eos_sequences: tp.List[str] = ["\sclass\s", "\sdef\s", "\s@", "<|endoftext|>", "<extra_id_0>"],
        model_kwargs: tp.Dict = {},
        generation_params: tp.Dict[str, tp.Any] = {},
        context_parser: BaseParser = TrivialContextParser(),
        add_extra_spaces_to_generation=0,
    ):
        """
        Class to generate code in fill-in-the-middle mode
        params:
            model_path: str - which model to use for generation, anything that can be passed to AutoModelForCausalLM.from_pretrained
            num_samples: int - number of samples to generate per task, values > 1 should be paired with generation_params
            prefix_tokens: tp.Union[str, tp.List[int]] = [] - tokens to insert before the left context. Can be either str or list of int tokens
            middle_tokens: tp.Union[str, tp.List[int]] = [] - tokens to insert before the right context (see Fill-In-the-Middle). Can be either str or list of int tokens
            suffix_tokens: tp.Union[str, tp.List[int]] = [] - tokens to insert after the right context (see Fill-In-the-Middle). Can be either str or list of int tokens
            max_context_length: int = None - truncation length for prompt, measured in tokens (len(left_context) + len(right_context) < max_context_length) 
            left_context_ratio: int = 1 - proportion of max_context_length given to left_context. 1 means 1:1 split between left and right, 3 means 3:1 split in favor of left context 
            dtype=torch.bfloat16 - torch dtype to use for inference
            eos_sequences: tp.List[str] = ["\sclass\s", "\sdef\s", "\s@", "<|endoftext|>", "<extra_id_0>"] - regular expressions that determine end of geneartion
            model_kwargs: tp.Dict = {} - kwargs to be passed to AutoModelForCausalLM.from_pretrained
            generation_params: tp.Dict[str, tp.Any] = {} - kwargs to be passed to AutoModelForCausalLM.generate
            context_parser: BaseParser = TrivialContextParser() - parser for left and right contexts
            add_extra_spaces_to_generation=0 - number of added extra spaces add the begining of generation to fix indentation. May be required due to bugs in some tokenizers (e.g. Codellama)
        """
        self.device = torch.device("cuda")
        # self.device = torch.device("cpu")
        logger.info(f"Loading model from {model_path} with kwargs f{model_kwargs}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=dtype, device_map="auto", trust_remote_code=True, **model_kwargs
        ).eval() 
        logger.info(f"Loaded model from {model_path} with kwargs f{model_kwargs}")
        logger.info(f"Device map: \n{self.model.hf_device_map}")

        self.num_samples = num_samples
        
        self.prefix_tokens = self.tokenize_special_tokens(prefix_tokens)
        self.middle_tokens = self.tokenize_special_tokens(middle_tokens)
        self.suffix_tokens = self.tokenize_special_tokens(suffix_tokens)

        logger.debug(f"prefix_tokens: {self.prefix_tokens}, middle_tokens: {self.middle_tokens}, suffix_tokens: {self.suffix_tokens}")

        self.eos_sequences = eos_sequences[:]

        #context truncation parameters
        self.max_context_length = max_context_length
        self.left_context_truncate_at = left_context_ratio / (left_context_ratio + 1)
        self.right_context_truncate_at = 1 / (left_context_ratio + 1)

        self.generation_params = generation_params
        self.generation_params['num_return_sequences'] = self.num_samples

        self.context_parser = context_parser
        # Number of tokens before and after truncating to max_context_length
        self.count_inferenced_tokens = []
        self.count_possible_tokens = []
        self.add_extra_spaces_to_generation = add_extra_spaces_to_generation

    def tokenize_special_tokens(self, str_or_list:  tp.Union[str, tp.List[int]]) -> torch.Tensor:        
        if type(str_or_list) == str:
            return self.tokenizer.encode(str_or_list, return_tensors="pt", add_special_tokens=False).to(self.device) # ['input_ids']
        else:
            return torch.as_tensor(str_or_list).unsqueeze(0).to(self.device)

    def _prepare_tokens(self, task: Task) -> torch.Tensor:
        left_context_str, right_context_str = self.context_parser.get_left_and_right_context(task)
        logger.info("\n" + "\n".join(left_context_str.split('\n')[-20:]))
        left_tokens = self.tokenizer.encode(
            left_context_str, return_tensors="pt", add_special_tokens=False).to(self.device) # ['input_ids']
        right_tokens = self.tokenizer.encode(
            right_context_str, return_tensors="pt", add_special_tokens=False).to(self.device) # ['input_ids']
        self.count_possible_tokens.append(left_tokens.shape[1] + right_tokens.shape[1])
        if self.max_context_length and left_tokens.shape[1] + right_tokens.shape[1] > self.max_context_length:
            logger.debug("Truncating context")
            
            left_tokens = left_tokens[:, -min(int(self.max_context_length * self.left_context_truncate_at), left_tokens.shape[1]) + 1:]
            right_tokens = right_tokens[:, :min(int(self.max_context_length * self.right_context_truncate_at), right_tokens.shape[1]) - 1]
        tokens = torch.cat([self.prefix_tokens, left_tokens, self.middle_tokens, right_tokens, self.suffix_tokens], dim=-1).type(torch.long)
        return tokens
    
    def _postprocess(self, generation: str):
        new_gen = []
        for i, line in enumerate(generation.split('\n')):
            if i == 0 and self.add_extra_spaces_to_generation: 
                # ugly hack for codellama, weirdly removing space for skip_special_tokens=True
                line = ' '*self.add_extra_spaces_to_generation + line
            for eos in self.eos_sequences:
                if re.search(eos, line):
                    return "\n".join(new_gen).rstrip() + '\n\n'
            new_gen.append(line)
        return "\n".join(new_gen).rstrip() + '\n\n'

    @torch.no_grad()
    def generate(self, tasks: tp.List[Task]) -> tp.List[tp.List[str]]:
        res = []
        for i, task in tqdm(enumerate(tasks)):
            tokens = self._prepare_tokens(task)
            if i == 0:
                logger.debug(f"\nTokens: {tokens[:, :5]} ... {tokens[:, -5:]}\n")
            generated_tokens = self.model.generate(tokens, **self.generation_params)
            generations = self.tokenizer.batch_decode(generated_tokens[:, tokens.shape[1]:], skip_special_tokens=True)
            if i % 1 == 0:
                logger.debug(f"Generation for task {i}:\n{self._postprocess(generations[0])}")
            res.append([self._postprocess(t) for t in generations])
            self.count_inferenced_tokens.append([len(t) for t in tokens])
        return res


class LMGenerator(InfillGenerator):
    def __init__(self, 
        lm_prefix_tokens: tp.Union[str, tp.List[int]] = [],
        lm_suffix_tokens: tp.Union[str, tp.List[int]] = [],
        **kwargs
    ):
        """
        Class to generate code in causal LM mode, uses only left context
        params:
            lm_prefix_tokens: tp.Union[str, tp.List[int]] = [] - tokens to insert before the context. Can be either str or list of int tokens
            lm_suffix_tokens: tp.Union[str, tp.List[int]] = [] - tokens to insert after the context. Can be either str or list of int tokens
        """
        super().__init__(**kwargs)
        self.lm_prefix_tokens = super().tokenize_special_tokens(lm_prefix_tokens)
        self.lm_suffix_tokens = super().tokenize_special_tokens(lm_suffix_tokens)
        logger.debug(f"lm_prefix_tokens: {self.lm_prefix_tokens}, lm_suffix_tokens: {self.lm_suffix_tokens}")

    def _prepare_tokens(self, task: Task) -> torch.Tensor:
        left_context_str, _ = self.context_parser.get_left_and_right_context(task)
        logger.info("\n" + "\n".join(left_context_str.split('\n')[-20:]))
        left_tokens = self.tokenizer.encode(
            left_context_str, return_tensors="pt", add_special_tokens=False).to(self.device) # ['input_ids']
        self.count_possible_tokens.append(left_tokens.shape[1])
        if self.max_context_length and left_tokens.shape[1] > self.max_context_length:
            left_tokens = left_tokens[:, -self.max_context_length:]
        tokens = torch.cat([self.lm_prefix_tokens, left_tokens, self.lm_suffix_tokens], dim=-1).type(torch.long)
        return tokens


    
