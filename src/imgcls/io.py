from pathlib import Path

__all__ = [
    'CACHE_DIRECTORY',
    'TRAIN_DIR',
    'TEST_DIR',
]

CACHE_DIRECTORY = Path.home() / '.cache' / 'comvis' / 'imgcls'
TRAIN_DIR = CACHE_DIRECTORY / 'train'
TEST_DIR = CACHE_DIRECTORY / 'test'
