from random import sample, seed
from shutil import copy
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("-d", "--input_dir", type=str, default="data/datasets/main/test", help="Input directory")
parser.add_argument("-n", "--output_dir", type=str, default="data/datasets/toy/test", help="Output directory")
parser.add_argument("-k", "--k_files", type=int, default=50, help="Number of files to copy")
parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing toy dataset")
args = parser.parse_args()

seed(0)

def select_and_copy_files(src_dir, dst_dir, num_files=50):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # Ensure the source directory exists
    if not src_path.exists() or not src_path.is_dir():
        print(f"Source directory {src_dir} does not exist or is not a directory.")
        return

    # Ensure the destination directory exists, create it if it doesn't
    if not dst_path.exists():
        dst_path.mkdir(parents=True)
        print(f"Destination directory {dst_dir} created.")

    # Get a list of files in the source directory
    files = [f for f in src_path.iterdir() if f.is_file()]

    # Ensure there are enough files to select from
    if len(files) < num_files:
        print(f"Not enough files in {src_dir} to select {num_files} files.")
        return

    # Select random files
    selected_files = sample(files, num_files)

    # Copy selected files to the destination directory
    for file in selected_files:
        copy(file, dst_path / file.name)

    print(f"Successfully copied {num_files} files from {src_dir} to {dst_dir}.")

def copy_existing_files(src_dir, dst_dir):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # Ensure the source directory exists
    if not src_path.exists() or not src_path.is_dir():
        print(f"Source directory {src_dir} does not exist or is not a directory.")
        return

    # Ensure the destination directory exists, create it if it doesn't
    if not dst_path.exists():
        dst_path.mkdir(parents=True)
        print(f"Destination directory {dst_dir} created.")

    # Get a list of file names in the destination directory
    for f in dst_path.iterdir():
        if f.is_file() and (src_path / f.name).is_file():
            copy(src_path / f.name, f)

    print(f"Successfully copied relevant files from {src_dir} to {dst_dir}.")

if args.overwrite:
    select_and_copy_files(args.input_dir, args.output_dir, num_files=args.k_files)
else:
    copy_existing_files(args.input_dir, args.output_dir)