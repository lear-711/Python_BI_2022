#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='+')
args = parser.parse_args()

with open(args.file[0], 'r') as file:
    lines = file.readlines()
    lines.sort()
for each in lines:
    print(each)


