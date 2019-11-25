from pathlib import Path
import pickle


BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data'
RESULTS_BASE_DIR = BASE_DIR / 'results'


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
