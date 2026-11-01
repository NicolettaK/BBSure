"""
Data directory of BBSure: the environment variable BBSURE_DATA, or the
data/ directory of the repository. Configs refer to it as @DATA@.
"""
import os

DATA_DIR = os.environ.get(
    'BBSURE_DATA',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'data'))


def load_yaml(fname):
    """yaml file with @DATA@ replaced by DATA_DIR."""
    import yaml
    return yaml.safe_load(open(fname).read().replace('@DATA@', DATA_DIR))
