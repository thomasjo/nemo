import random
import shutil
from contextlib import contextmanager
from pathlib import Path

LABEL_PREFIXES = {
    "Bent-": "benthic",
    "Plank-": "planktic",
    "SedA-": "sediment",
    "SedB-": "sediment",
}

TEST_FILE_PREFIXES = [
    "Bent-01",
    # "Bent-04",
    # "Bent-09",
    # "Plank-04",
    "Plank-09",
    "SedA-07",
    "SedB-02",
]


@contextmanager
def random_seed(seed):
    old_state = random.getstate()
    yield random.seed(seed)
    random.setstate(old_state)


def recreate_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)

    return path


def label_from_name(name):
    for prefix, label in LABEL_PREFIXES.items():
        if name.startswith(prefix):
            return label
    return None


def is_test_file(name):
    for prefix in TEST_FILE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def create_validation_set(source_dir, *, ratio):
    source_files = sorted([file for file in source_dir.rglob("*.png")])

    # Make the splitting reproducible by using a fixed seed.
    with random_seed(42):
        random.shuffle(source_files)
        split = round((1.0 - ratio) * len(source_files))
        valid_files = source_files[split:]

    for image_file in valid_files:
        target_dir = valid_dir / image_file.parent.name
        shutil.move(str(image_file), str(target_dir))


if __name__ == "__main__":
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent.parent

    # TODO: Make these configurable?
    data_dir = root_dir / "data"
    processed_dir = data_dir / "processed"

    train_dir = recreate_dir(data_dir / "train")
    valid_dir = recreate_dir(data_dir / "valid")
    test_dir = recreate_dir(data_dir / "test")

    for label in LABEL_PREFIXES.values():
        (train_dir / label).mkdir(exist_ok=True)
        (valid_dir / label).mkdir(exist_ok=True)
        (test_dir / label).mkdir(exist_ok=True)

    all_patch_images = sorted(processed_dir.rglob("*-patch*.png"))
    for image_file in all_patch_images:
        image_name = image_file.name
        image_label = label_from_name(image_name)
        if image_label is None:
            continue

        target_dir = test_dir if is_test_file(image_name) else train_dir
        target_file = target_dir / image_label / image_name

        shutil.copy2(image_file, target_file)

    # Extract a validation set from 15% of the training set.
    create_validation_set(train_dir, ratio=0.15)
