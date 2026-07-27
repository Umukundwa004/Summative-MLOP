import os
import sys

# Define required structure
REQUIRED_SPLITS = ['train', 'test']
REQUIRED_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def validate_dataset(data_dir="data"):
    print(f"\n🔍 Validating dataset structure in: '{data_dir}/'\n")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Root directory '{data_dir}' does not exist!")
        return False

    errors = []
    warnings = []
    summary = {}

    for split in REQUIRED_SPLITS:
        split_path = os.path.join(data_dir, split)
        summary[split] = {}

        if not os.path.exists(split_path):
            errors.append(f"Missing main folder: '{split_path}'")
            continue

        for cls in REQUIRED_CLASSES:
            class_path = os.path.join(split_path, cls)

            if not os.path.exists(class_path):
                errors.append(f"Missing class folder: '{class_path}'")
                summary[split][cls] = 0
                continue

            # Count valid image files
            files = os.listdir(class_path)
            valid_images = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
            invalid_files = len(files) - len(valid_images)

            summary[split][cls] = len(valid_images)

            if len(valid_images) == 0:
                warnings.append(f"Folder is empty: '{class_path}'")

            if invalid_files > 0:
                warnings.append(f"Found {invalid_files} non-image/unsupported files in '{class_path}'")

    # Print Summary Table
    print(f"{'Split':<10} | {'Class':<12} | {'Image Count':<12}")
    print("-" * 40)
    for split, classes in summary.items():
        for cls, count in classes.items():
            print(f"{split:<10} | {cls:<12} | {count:<12}")
    print("-" * 40)

    # Print Diagnostic Results
    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n❌ STRUCTURAL ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        print("\n❌ Dataset validation FAILED.")
        return False
    
    print("\n✅ Dataset structure is valid and ready for training!\n")
    return True

if __name__ == "__main__":
    is_valid = validate_dataset()
    if not is_valid:
        sys.exit(1)