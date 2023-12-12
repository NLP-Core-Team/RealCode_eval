from dataclasses import dataclass
import typing as tp
from collections import deque, namedtuple
import re
import ast
from pathlib import Path

from .datatypes import Task
from transformers import AutoTokenizer

    
Import = namedtuple("Import", ["module", "name", "alias"])

"""
>>> Imports
import math
<<< imports

>>> file scope
def get_c():
 	return 1

<<< file scope
>>> outer scope
class Foo:
 	def __init__(self, a):
		self.a = a
<<< outer scope
>>> inner scope
    @staticmethod
	def bar():
		'''
		Turn Foo into bar
		'''
<<< inner scope
>>> body (unavailable for model)
        bar = 'B'
        self.a = bar
        return self
<<< body (unavailable for model)
>>> outer scope
    def bar2():
     	self.a = 'C'
     	return self
<<< outer scope
>>> file scope
class Foo2:
	...

<<< file scope
"""

@dataclass(frozen=False)
class ParsedContext:
    imports = ''
    file = ''
    outer = ''
    inner = ''
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __str__(self):
        return (

            '\n----- imports -----\n' +
            self.imports +
            '\n----- end imports -----\n' +
            '\n----- file -----\n' +
            (
                ('\n'.join(self.file.split('\n')[:10]) + '\n...\n' + '\n'.join(self.file.split('\n')[-15:])) if len(self.file.split('\n')) > 20 else self.file
            ) + 
            '\n----- end file -----\n' +
            '\n----- outer -----\n' +
            (
                ('\n'.join(self.outer.split('\n')[:15]) + '\n...\n' + '\n'.join(self.outer.split('\n')[-15:])) if len(self.outer.split('\n')) > 20 else self.outer
            ) + 
            '\n----- end outer -----\n' +
            '\n----- inner -----\n' +
            '\n'.join(self.inner.split('\n')) + 
            '\n----- end inner -----\n' 

        )


def get_indent(code):
    line = code.split('\n')[0]
    return len(line) - len(line.strip())


def parse_context(context: str, indent: int, side: tp.Literal['left', 'right']) -> ParsedContext:
    res = ParsedContext()
    if side == 'left':
        cur_scope = deque()
        state = 'inner'

        for line in reversed(context.split('\n')):
            if line.startswith('import') or (line.startswith('from') and ' import ' in line):
                res['imports'] += line + '\n' 
                continue

            if state == 'inner_wait@':
                if not line.lstrip().startswith('@'):
                    res['inner'] = "\n".join(cur_scope)
                    cur_scope = deque()
                    if indent > 0: 
                        state = 'outer'
                    else:
                        state = 'file'

            cur_scope.appendleft(line)
            if state == 'inner':
                if line.strip().startswith('def '):
                    state = 'inner_wait@'
            if state == 'outer':
                if line.startswith('class'):
                    res['outer'] = "\n".join(cur_scope)
                    state = 'file'
                    cur_scope = deque()
        if state == 'inner_wait@':
            state = 'inner'
        res[state] = "\n".join(cur_scope)
    elif side == 'right':
        cur_scope = deque()
        state = 'outer'

        for line in context.split('\n'):
            if state == 'outer':
                if (
                    line.strip()
                    and not line.startswith(' ')
                ):
                    res['outer'] = "\n".join(cur_scope)
                    state = 'file'
                    cur_scope = deque()
            cur_scope.append(line)
        res[state] = "\n".join(cur_scope)
    return res


class BaseParser:
    def get_left_and_right_context(self, task: Task) -> tp.Tuple[str, str]:
        """
        main method, that returns tuple (left_context, right_context) for the task
        """
        raise NotImplementedError()


class TrivialContextParser(BaseParser):
    def get_left_and_right_context(self, task: Task) -> tp.Tuple[str, str]:
        """
        returns left and right context without processing
        """
        return task.left_context, task.right_context


class SmartContextParser(BaseParser):
    def __init__(self, 
        left_config = ['imports', 'file', 'outer', 'inner'],
        right_config = ['outer', 'file']          
    ):
        self.left_config = left_config
        self.right_config = right_config

    def get_left_and_right_context(self, task: Task) -> tp.Tuple[str, str]:
        """
        
        """
        indent = (len(task.gt) - len(task.gt.lstrip()))
        left_context_parsed = parse_context(task.left_context, indent, 'left')
        left_context = "\n".join([left_context_parsed[k] for k in self.left_config])
        right_context_parsed = parse_context(task.right_context, indent, 'right')   
        right_context = "\n".join([right_context_parsed[k] for k in self.right_config])
        return left_context, right_context
    
class ImportResolutionParser(BaseParser):
    def __init__(self,
        data_root: str,
        left_config = ['imports', 'file', 'outer', 'inner'],
        right_config = ['outer', 'file']          
    ):
        """

        """
        self.data_root = data_root
        self.left_config = left_config
        self.right_config = right_config

    def _desc_func(self, functionNode, lines):
        return " ".join([t.strip() for t in lines[functionNode.lineno-1: functionNode.body[0].lineno - 1]])

    def _parse_file(self, filename, func_names):
        ans = []
        with open(filename, 'r', encoding='UTF-8') as f:
            text = f.read()
            lines = text.split('\n')
            node = ast.parse(text)
        if func_names:
            functions = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in func_names]
            classes = [n for n in node.body if isinstance(n, ast.ClassDef) and n.name in func_names]
        else:
            functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

        for function in functions:
            s = self._desc_func(function, lines)
            ans.append('' + s)

        for class_ in classes:
            ans.append("class " + class_.name)
            methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
            for method in methods:
                s = self._desc_func(method, lines)
                ans.append('    ' + s)
        return "\n".join(ans)

    def _get_imports(self, code):     
        root = ast.parse(code)

        for node in ast.iter_child_nodes(root):
            if isinstance(node, ast.Import):
                module = [t.name for t in node.names]
                yield (
                    Import(module, [], []), 
                    " ".join(code.split('\n')[node.lineno-1: node.end_lineno])
                )
            elif isinstance(node, ast.ImportFrom):  
                module = node.module.split('.')
                yield (
                    Import(module, [n.name for n in node.names], [n.name for n in node.names]), 
                    " ".join(code.split('\n')[node.lineno-1: node.end_lineno])
                )
            else:
                continue
    
    def _resolve_imports(self, task: Task) -> str:
        repo = (Path(self.data_root) / task.repo).resolve()
        ans = []
        for imp, line in self._get_imports(task.left_context):
            pth = repo / ("/".join(imp.module) + '.py')
            if imp.module and pth.exists():
                ans.append(line)
                ans.append(self._parse_file(pth, imp.name))
            else:
                ans.append(line)
        return '\n'.join(ans)
        
    def get_left_and_right_context(self, task: Task) -> tp.Tuple[str, str]:
        indent = (len(task.gt) - len(task.gt.lstrip()))
        left_context_parsed = parse_context(task.left_context, indent, 'left')
        left_context = "\n".join([
            left_context_parsed[k] if k != 'imports' else self._resolve_imports(task) + '\n'
            for k in self.left_config
        ])
        right_context_parsed = parse_context(task.right_context, indent, 'right')   
        right_context = "\n".join([right_context_parsed[k] for k in self.right_config])
        return left_context, right_context


class ImportCopyParser(ImportResolutionParser):
    def _parse_file(self, filename, func_names):
        ans = []
        with open(filename, 'r', encoding='UTF-8') as f:
            text = f.read()
            lines = text.split('\n')
            node = ast.parse(text)
        if func_names:
            functions = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name not in func_names and n.col_offset == 0]
            classes = [n for n in node.body if isinstance(n, ast.ClassDef) and n.name not in func_names and n.col_offset == 0]
            skip_intervals = [(t.lineno-1, t.end_lineno-1) for t in functions + classes]
            skip_intervals.sort()
        else:
            functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
            skip_intervals = []
        interval_id = 0
        i = 0
        while i < len(lines):
            if interval_id < len(skip_intervals) and i >= skip_intervals[interval_id][0]:
                i = skip_intervals[interval_id][1]
                interval_id += 1
            else:
                ans.append(lines[i])
                i += 1
        return "\n".join(ans)
    
    def _resolve_imports(self, task: Task) -> str:
        repo = (Path(self.data_root) / task.repo).resolve()
        ans = []
        for imp, line in self._get_imports(task.left_context):
            module_pth = ("/".join(imp.module) + '.py')
            pth = repo / module_pth
            if imp.module and pth.exists():
                ans.append('#' + module_pth)
                ans.append(self._parse_file(pth, imp.name))

        cur_module = task.path_from_root.replace('/', '.').replace('.py', '')
        for file in [
            f for f in repo.rglob('*.py') 
            if {"venv_bench", '.ipynb_checkpoints'}.isdisjoint(set([str(p) for p in f.parts]))
        ]:
            file = file.absolute()
            with open(file, 'r', encoding='UTF-8') as f:
                text = f.read()
            if cur_module in text:
                ans.append('#' + str(file.relative_to(repo)))
                ans.append(text)
        ans.append('#' + task.path_from_root)
        for imp, line in self._get_imports(task.left_context):
            ans.append(line)
        return '\n'.join(ans)
