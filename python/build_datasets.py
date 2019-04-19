import shutil

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


if __name__ == "__main__":
    file_dir: Path = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    data_dir = root_dir / "data"
    processed_dir = data_dir / "processed"

    train_dir = recreate_dir(data_dir / "train")
    test_dir = recreate_dir(data_dir / "test")

    for label in LABEL_PREFIXES.values():
        (train_dir / label).mkdir(exist_ok=True)
        (test_dir / label).mkdir(exist_ok=True)

    all_patch_images = sorted(processed_dir.rglob("*-patch*.png"))
    for image_file in all_patch_images:
        print(image_file)
        image_name = image_file.name
        image_label = label_from_name(image_name)
        if image_label is None:
            continue

        target_dir = test_dir if is_test_file(image_name) else train_dir
        target_file = target_dir / image_label / image_name

        shutil.copy2(image_file, target_file)
