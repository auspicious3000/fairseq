
import os
import itertools
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv')
    parser.add_argument('--output-path')
    args = parser.parse_args()

    tsv = args.tsv
    output_path = args.output_path
    # dirname, basename = os.path.split(tsv)
    # split, ext = os.path.splitext(basename)
    with open(tsv, 'r') as ftsv, open(output_path, 'w') as fltr:
        root_dir = ftsv.readline().strip()
        for l in ftsv:
            fltr.write('A\n')
