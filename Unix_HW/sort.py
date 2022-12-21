#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='*')
args = parser.parse_args()

lines = []
if len(args.file) == 0:
    for line in sys.stdin:
        lines.append(line)
else:
    with open(args.file[0], 'r') as file:
        lines = file.readlines()
lines.sort()
print("".join(lines))


