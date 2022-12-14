#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='*')
parser.add_argument('-n', '--numbers', action='store')

args = parser.parse_args()


if args.numbers:
    N = int(args.numbers)
else:
    N = 10

if len(args.file) == 0:
    all_lines = []
    for line in sys.stdin:
        all_lines.append(line)
    for line_2 in (all_lines[-N:]):
        print(line_2, end ='')
else:
    with open(args.file[0], 'r') as file:
        lines = file.readlines()
        for line in (lines[-N:]):
            print(line, end ='')


