import random
import shutil
from contextlib import contextmanager
from pathlib import Path

LABEL_PREFIXES = {
    "Bent": "benthic",
    "Plank": "planktic",
    "Sed": "sediment",
}


@contextmanager
def random_seed(seed):
    old_state = random.getstate()
    yield random.seed(seed)
    random.setstate(old_state)


def recreate_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)

    return path


def split_set(input, ratio):
    split_idx = round(len(input) * ratio)
    a, b = input[:split_idx], input[split_idx:]
    return a, b


def copy_files_to_dir(src_files, dst_dir):
    dst_dir.mkdir(exist_ok=True)
    for src_file in src_files:
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent.parent

    # TODO: Make these configurable?
    data_dir = root_dir / "data"
    processed_dir = data_dir / "processed"

    train_dir = recreate_dir(data_dir / "train")
    valid_dir = recreate_dir(data_dir / "valid")
    test_dir = recreate_dir(data_dir / "test")

    for prefix, label in LABEL_PREFIXES.items():
        # Find all patches for current label.
        patches = sorted(processed_dir.rglob(f"{prefix}*patch*.png"))

        # Shuffle patches in a reproducible manner.
        with random_seed(42):
            random.shuffle(patches)

        # Split label patches into subsets.
        train_patches, test_patches = split_set(patches, 0.8)
        valid_patches, test_patches = split_set(test_patches, 0.5)

        # Copy all image patches to subset directories.
        copy_files_to_dir(train_patches, train_dir / label)
        copy_files_to_dir(valid_patches, valid_dir / label)
        copy_files_to_dir(test_patches, test_dir / label)
