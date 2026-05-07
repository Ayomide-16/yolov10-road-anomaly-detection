"""
Dataset Preparation Script for MCE 415 - Road Anomaly Detection (Group 8: YOLOv10)
==================================================================================

This script merges three separate datasets (Pothole, Speedbumps, Cracks) into a 
single unified dataset suitable for YOLO training. It handles:
  - Class remapping to a unified scheme: {0: Pothole, 1: Speedbump, 2: Crack}
  - Filtering unwanted classes from the Speedbumps dataset
  - Converting polygon (segmentation) labels to bounding boxes for Cracks
  - Merging everything into one dataset with a 70/15/15 train/val/test split
  - Validating the final dataset for orphaned files and corrupt entries

Usage:
    python prepare_dataset.py

Output:
    unified_dataset/
    ├── train/images/  train/labels/
    ├── val/images/    val/labels/
    ├── test/images/   test/labels/
    └── data.yaml

Author: Group 8 - MCE 415 (FUT Minna)
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict


# -- Configuration -----------------------------------------------------------------

# Set a fixed seed so the split is reproducible every time you run this script.
# Reproducibility matters because your teammates and evaluators should get the
# exact same split if they re-run this.
RANDOM_SEED = 42

# The three unified class names and their IDs.
# Every label file we produce will use these IDs.
CLASS_MAP = {
    0: "Pothole",
    1: "Speedbump",
    2: "Crack",
}

# Where the raw downloaded datasets live (relative to this script).
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "Downloaded Datasets"
OUTPUT_DIR = BASE_DIR / "unified_dataset"

# Split ratios -- must sum to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# -- Helper Functions --------------------------------------------------------------

def polygon_to_bbox(coords):
    """
    Convert a list of polygon coordinates to a YOLO bounding box.
    
    Polygon format from the label: [x1, y1, x2, y2, x3, y3, ...]
    All values are normalized (0-1 range relative to image dimensions).
    
    YOLO bbox format: (x_center, y_center, width, height), also normalized.
    
    The conversion simply finds the min/max of all x and y coordinates
    to produce an axis-aligned bounding box that encloses the polygon.
    """
    # Separate x and y coordinates (even indices = x, odd indices = y)
    x_coords = [coords[i] for i in range(0, len(coords), 2)]
    y_coords = [coords[i] for i in range(1, len(coords), 2)]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # YOLO format: center_x, center_y, width, height
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def process_pothole_label(label_path):
    """
    Process a Pothole label file.
    
    The Pothole dataset already uses class 0, which matches our unified scheme.
    Labels are already in YOLO bbox format, so we just pass them through as-is.
    
    Returns a list of label lines (strings) in YOLO format, or None if the file
    is empty or invalid.
    """
    output_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Pothole labels should have exactly 5 parts: class x y w h
            if len(parts) != 5:
                continue
            # Keep class as 0 (Pothole) -- no remapping needed
            output_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    return output_lines if output_lines else None


def process_speedbump_label(label_path):
    """
    Process a Speedbumps label file.
    
    The Speedbumps dataset has 5 classes:
      0: Manhole, 1: Open Manhole, 2: Pothole, 3: Speed Bump, 4: Unmarked Bump
    
    We ONLY keep classes 3 (Speed Bump) and 4 (Unmarked Bump), and remap both
    to our unified class 1 (Speedbump). All other classes are discarded since
    we already have a dedicated Pothole dataset and don't need Manhole data.
    
    Returns a list of label lines, or None if no relevant annotations remain.
    """
    # The class IDs we want from this dataset
    KEEP_CLASSES = {3, 4}

    output_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            
            original_class = int(parts[0])
            if original_class not in KEEP_CLASSES:
                continue  # skip Manhole, Open Manhole, Pothole from this dataset

            # Both Speed Bump and Unmarked Bump become class 1 (Speedbump)
            output_lines.append(f"1 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    return output_lines if output_lines else None


def process_crack_label(label_path):
    """
    Process a Cracks label file.
    
    The Cracks dataset uses POLYGON (segmentation) format rather than bounding boxes.
    Each line looks like: class_id x1 y1 x2 y2 x3 y3 ... xN yN
    
    We convert each polygon to an axis-aligned bounding box by finding the
    min/max of the x and y coordinates. The class is remapped to 2 (Crack).
    
    Returns a list of label lines in YOLO bbox format, or None if empty.
    """
    output_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue  # need at least class + 2 points (5 values)

            # Extract the polygon coordinates (everything after the class ID)
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                continue  # skip malformed lines

            # Must have an even number of coordinates (x,y pairs)
            if len(coords) % 2 != 0:
                continue

            x_center, y_center, width, height = polygon_to_bbox(coords)

            # Clamp values to valid range [0, 1] as a safety measure
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.001, min(1.0, width))
            height = max(0.001, min(1.0, height))

            output_lines.append(
                f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

    return output_lines if output_lines else None


def collect_image_label_pairs(image_dir, label_dir, process_fn, source_name):
    """
    Walk through an image directory and its corresponding label directory,
    process each label with the given function, and collect valid pairs.
    
    Args:
        image_dir:   Path to the folder containing .jpg images
        label_dir:   Path to the folder containing .txt label files
        process_fn:  Function that processes a single label file
        source_name: A tag for logging (e.g. "Pothole", "Speedbumps")
    
    Returns:
        List of tuples: (image_path, processed_label_lines, source_name)
    """
    pairs = []
    skipped = 0
    
    if not image_dir.exists():
        print(f"  [WARN] Image directory not found: {image_dir}")
        return pairs

    if not label_dir.exists():
        print(f"  [WARN] Label directory not found: {label_dir}")
        return pairs

    # Build a set of available label files for quick lookup
    label_files = {f.stem: f for f in label_dir.glob("*.txt")}

    for img_file in sorted(image_dir.glob("*.jpg")):
        stem = img_file.stem
        if stem not in label_files:
            skipped += 1
            continue

        # Process the label through the appropriate handler
        processed = process_fn(label_files[stem])
        if processed is None:
            skipped += 1
            continue  # no valid annotations after filtering

        pairs.append((img_file, processed, source_name))

    print(f"  [{source_name}] Collected {len(pairs)} valid pairs, skipped {skipped}")
    return pairs


def create_output_dirs():
    """
    Create the unified dataset directory structure.
    Wipes any existing output to ensure a clean state.
    """
    if OUTPUT_DIR.exists():
        print(f"  Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"  Created output structure at: {OUTPUT_DIR}")


def write_data_yaml():
    """
    Generate the data.yaml file that YOLO requires to locate the dataset.
    
    The paths are relative so that this works both locally and on Kaggle
    after uploading the unified_dataset folder.
    """
    data = {
        "path": ".",               # dataset root (will be overridden in the notebook)
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 3,
        "names": ["Pothole", "Speedbump", "Crack"],
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"  Written data.yaml -> {yaml_path}")


def validate_dataset():
    """
    Run post-split validation checks on the unified dataset:
      1. Every image has a corresponding label file
      2. Every label file has a corresponding image
      3. All label lines are well-formed (5 space-separated values)
      4. Report class distribution per split
    
    This catches problems BEFORE you spend hours training on Kaggle.
    """
    print("\n=== VALIDATION ===")
    all_good = True

    for split in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / split / "images"
        lbl_dir = OUTPUT_DIR / split / "labels"

        img_stems = {f.stem for f in img_dir.glob("*.jpg")}
        lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}

        # Check for orphaned images (no label)
        orphan_images = img_stems - lbl_stems
        if orphan_images:
            print(f"  [WARN] {split}: {len(orphan_images)} images without labels")
            all_good = False

        # Check for orphaned labels (no image)
        orphan_labels = lbl_stems - img_stems
        if orphan_labels:
            print(f"  [WARN] {split}: {len(orphan_labels)} labels without images")
            all_good = False

        # Count class distribution
        class_counts = defaultdict(int)
        total_annotations = 0
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"  [WARN] Malformed line in {lbl_file.name}: {line[:50]}")
                        all_good = False
                        continue
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
                    total_annotations += 1

        print(f"\n  [{split.upper()}] {len(img_stems)} images, {total_annotations} annotations")
        for cls_id in sorted(class_counts.keys()):
            cls_name = CLASS_MAP.get(cls_id, f"Unknown({cls_id})")
            print(f"    Class {cls_id} ({cls_name}): {class_counts[cls_id]} annotations")

    if all_good:
        print("\n  [OK] All validation checks passed!")
    else:
        print("\n  [!!] Some issues found -- review warnings above.")


# -- Main Pipeline -----------------------------------------------------------------

def main():
    print("=" * 70)
    print("MCE 415 - Dataset Preparation (Group 8: YOLOv10)")
    print("=" * 70)

    # Seed for reproducibility
    random.seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # STEP 1: Collect all image-label pairs from each dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Collecting image-label pairs from all datasets...\n")

    all_pairs = []

    # -- Pothole dataset (all splits) --
    print("  Processing Pothole dataset...")
    for split_name, split_folder in [("train", "train"), ("valid", "valid")]:
        img_dir = RAW_DATA_DIR / "Pothole" / split_folder / "images"
        lbl_dir = RAW_DATA_DIR / "Pothole" / split_folder / "labels"
        pairs = collect_image_label_pairs(img_dir, lbl_dir, process_pothole_label, "Pothole")
        all_pairs.extend(pairs)

    # -- Speedbumps dataset (train only -- no valid/test splits exist) --
    print("\n  Processing Speedbumps dataset...")
    img_dir = RAW_DATA_DIR / "Speedbumps" / "train" / "images"
    lbl_dir = RAW_DATA_DIR / "Speedbumps" / "train" / "labels"
    pairs = collect_image_label_pairs(img_dir, lbl_dir, process_speedbump_label, "Speedbumps")
    all_pairs.extend(pairs)

    # -- Cracks dataset (all splits) --
    print("\n  Processing Cracks dataset...")
    for split_name, split_folder in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        img_dir = RAW_DATA_DIR / "Cracks" / split_folder / "images"
        lbl_dir = RAW_DATA_DIR / "Cracks" / split_folder / "labels"
        pairs = collect_image_label_pairs(img_dir, lbl_dir, process_crack_label, "Cracks")
        all_pairs.extend(pairs)

    print(f"\n  TOTAL collected: {len(all_pairs)} image-label pairs")

    # -------------------------------------------------------------------------
    # STEP 2: Shuffle and split into 70/15/15
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Shuffling and splitting (70/15/15)...\n")

    random.shuffle(all_pairs)

    total = len(all_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }

    for split_name, split_data in splits.items():
        # Count per-source distribution to verify balance
        source_counts = defaultdict(int)
        for _, _, source in split_data:
            source_counts[source] += 1
        source_str = ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))
        print(f"  {split_name}: {len(split_data)} samples ({source_str})")

    # -------------------------------------------------------------------------
    # STEP 3: Create output directories and copy files
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Creating output directories and writing files...\n")

    create_output_dirs()

    # Track how many files we write per split for a sanity check
    write_counts = defaultdict(int)

    for split_name, split_data in splits.items():
        img_out_dir = OUTPUT_DIR / split_name / "images"
        lbl_out_dir = OUTPUT_DIR / split_name / "labels"

        for idx, (img_path, label_lines, source) in enumerate(split_data):
            # To avoid filename collisions between datasets (e.g. both Pothole
            # and Cracks might have a file called "100_jpg.rf....txt"), we prefix
            # each filename with its source dataset tag.
            source_prefix = source.lower().replace(" ", "")
            new_stem = f"{source_prefix}_{img_path.stem}"

            # Copy image
            dst_img = img_out_dir / f"{new_stem}.jpg"
            shutil.copy2(img_path, dst_img)

            # Write processed label
            dst_lbl = lbl_out_dir / f"{new_stem}.txt"
            with open(dst_lbl, "w") as f:
                f.write("\n".join(label_lines) + "\n")

            write_counts[split_name] += 1

    for split_name, count in write_counts.items():
        print(f"  Written {count} pairs to {split_name}/")

    # -------------------------------------------------------------------------
    # STEP 4: Generate data.yaml
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Generating data.yaml...\n")
    write_data_yaml()

    # -------------------------------------------------------------------------
    # STEP 5: Validate the final dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Running validation checks...")
    validate_dataset()

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total samples: {total} (train: {len(splits['train'])}, "
          f"val: {len(splits['val'])}, test: {len(splits['test'])})")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Zip the 'unified_dataset' folder")
    print("  2. Upload it to Kaggle as a dataset")
    print("  3. Run the training notebook on Kaggle")


if __name__ == "__main__":
    main()
