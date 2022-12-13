#!/usr/bin/env python
# coding: utf-8

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='*')

args = parser.parse_args()

all_lines = []
with open(args.file[0], 'r') as file:
    for line in file.readlines():
        all_lines.append(line)
unique_lines = set(all_lines)

with open(args.file[1], 'w') as file:
    file.writelines(unique_lines)





