#!/usr/bin/env python3

import os
import glob

input_dir = "xyz.original"
output_dir = "xyz"

os.makedirs(output_dir, exist_ok=True)

xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))

for xyz_file in xyz_files:
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    base = os.path.splitext(os.path.basename(xyz_file))[0]
    ext = '.xyz'
    i = 1
    idx = 0

    while idx < len(lines):
        while idx < len(lines) and not lines[idx].strip().isdigit():
            idx += 1
        if idx >= len(lines):
            break

        atom_count = int(lines[idx].strip())
        header = lines[idx + 1]
        atom_lines = lines[idx + 2 : idx + 2 + atom_count]

        out_name = os.path.join(output_dir, f"{base}_{i}{ext}")
        with open(out_name, 'w') as out:
            out.write(f"{atom_count}\n{header}")
            out.writelines(atom_lines)
        i += 1
        idx = idx + 2 + atom_count

    print(f"Processed {xyz_file} into {i-1} files.")