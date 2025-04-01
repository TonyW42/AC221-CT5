import argparse
import os
import shutil
import pandas as pd


def make_datasets(data_dir, max_people, erase_old):
    # Define path to the reference database directory
    reference_dir = os.path.join(data_dir, "reference_db")
    # Load the reference database CSV which contains the list of person names
    reference_db = pd.read_csv(os.path.join(reference_dir, "reference_db.csv"))
    
    # Randomly sample up to max_people entries from the reference database
    reference_db = reference_db.sample(n=max_people, random_state=42)

    # Define paths for the output template and probe databases
    template_dir = os.path.join(data_dir, "template_db")
    probe_dir = os.path.join(data_dir, "probe_db")

    # Optionally erase old directories if requested
    if erase_old:
        shutil.rmtree(template_dir, ignore_errors=True)
        shutil.rmtree(probe_dir, ignore_errors=True)

    # Create the template_db and probe_db directories if they don't exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(probe_dir, exist_ok=True)

    # Copy photos
    for name in reference_db["name"]:
        # Create subdirectories for each person inside template_db and probe_db
        os.makedirs(os.path.join(template_dir, name), exist_ok=True)
        os.makedirs(os.path.join(probe_dir, name), exist_ok=True)

        # Define source and destination paths for the template image
        template_src = os.path.join(reference_dir, name, f"{name}_0001.jpg")
        template_dst = os.path.join(template_dir, name, f"{name}_0001.jpg")
        # Define source and destination paths for the probe image
        probe_src = os.path.join(reference_dir, name, f"{name}_0002.jpg")
        probe_dst = os.path.join(probe_dir, name, f"{name}_0002.jpg")

        # Copy the images into their respective directories
        shutil.copy(template_src, template_dst)
        shutil.copy(probe_src, probe_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Path to the data directory")
    parser.add_argument("--max_people", type=int, default=100, help="Maximum number of distinct individuals to include in template_db")
    parser.add_argument("--erase_old", action="store_true", help="Whether to erase the existing template and probe db")
    args = parser.parse_args()

    make_datasets(args.data_dir, args.max_people, args.max_people)
