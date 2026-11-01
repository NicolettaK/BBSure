"""
Log of the decisions: every node of the tree with question, answer and
numbers, in decision_log.json and in readable form in decision_log.txt.
"""
import json
import os

import numpy as np


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


class DecisionLog:
    def __init__(self, runs_dir, verbose=True):
        self.events = []
        self.json_path = os.path.join(runs_dir, 'decision_log.json')
        self.txt_path = os.path.join(runs_dir, 'decision_log.txt')
        self.verbose = verbose
        self._lines = []

    def info(self, msg):
        if self.verbose:
            print(msg, flush=True)
        self._lines.append(msg)
        self._flush()

    def node(self, node, question, answer, **numbers):
        ev = dict(step=len(self.events) + 1, node=node, question=question,
                  answer=answer, **numbers)
        self.events.append(ev)
        nums = ', '.join(f"{k}={fmt(v)}" for k, v in numbers.items()
                         if not isinstance(v, (dict, list)))
        self.info(f"[{ev['step']:02d}] {node}: {question} -> {answer}"
                  + (f"   ({nums})" if nums else ''))

    def _flush(self):
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.events, f, indent=2, default=json_default)
            with open(self.txt_path, 'w') as f:
                f.write('\n'.join(self._lines) + '\n')
        except OSError:
            pass
