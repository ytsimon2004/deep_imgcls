from pathlib import Path

__all__ = [
    'TRAIN_DIR',
    'TEST_DIR',
]

CACHE_DIRECTORY = Path.home() / '.cache' / 'comvis' / 'imgcls'
TRAIN_DIR = CACHE_DIRECTORY / 'train'
TEST_DIR = CACHE_DIRECTORY / 'test'
