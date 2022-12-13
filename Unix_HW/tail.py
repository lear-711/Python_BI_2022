#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='*')
parser.add_argument('-n', '--numbers', action='store')

args = parser.parse_args()


if args.numbers:
    N = int(args.numbers)
else:
    N = 10

with open(args.file[0], 'r') as file:
    lines = file.readlines()
    for line in (lines[-N:]):
        print(line, end ='')


