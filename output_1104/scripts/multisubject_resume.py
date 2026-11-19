"""Preserve incomplete, explicitly recognized stage files before retrying."""
from pathlib import Path


def archive_file(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    archive=path.parent/'incomplete_attempts'
    archive.mkdir(exist_ok=True)
    number=1
    while (archive/f'attempt_{number:03d}').exists():
        number+=1
    destination=archive/f'attempt_{number:03d}'
    destination.mkdir()
    path.rename(destination/path.name)
    return destination/path.name


def archive_incomplete(directory, allowed_names):
    directory=Path(directory)
    allowed=set(allowed_names)
    existing=[p for p in directory.iterdir() if p.name not in ('.mapping.lock','incomplete_attempts')]
    unexpected=[p.name for p in existing if p.name not in allowed or not p.is_file()]
    if unexpected:
        raise ValueError('Unexpected files in incomplete output; preserve and inspect: '+str(unexpected))
    if not existing:
        return None
    archive=directory/'incomplete_attempts'
    archive.mkdir(exist_ok=True)
    number=1
    while (archive/f'attempt_{number:03d}').exists():
        number+=1
    destination=archive/f'attempt_{number:03d}'
    destination.mkdir()
    for path in existing:
        path.rename(destination/path.name)
    return destination
