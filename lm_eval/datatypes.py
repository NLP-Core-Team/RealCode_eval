from dataclasses import dataclass
import typing as tp

@dataclass(frozen=True)
class Task:
    repo: str
    repo_n: int
    path_from_root: str
    left_context: str
    right_context: str
    gt: str
    total_tests: int
    doc: str = ''


