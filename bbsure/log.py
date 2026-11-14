"""
Log of the decisions: every node of the tree with question, answer and
numbers, in decision_log.json and in readable form in decision_log.txt.
"""
import json
import os
import threading

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
        # phase 0 runs its two halves in threads that share this log
        self._lock = threading.RLock()

    def __getstate__(self):
        # the lock cannot be pickled (the profile scheme sends the log to
        # its worker processes): left out, a new one in the copy
        d = dict(self.__dict__)
        d.pop('_lock', None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._lock = threading.RLock()

    def info(self, msg):
        with self._lock:
            if self.verbose:
                print(msg, flush=True)
            self._lines.append(msg)
            self._flush()

    def node(self, node, question, answer, **numbers):
        with self._lock:
            ev = dict(step=len(self.events) + 1, node=node,
                      question=question, answer=answer, **numbers)
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
