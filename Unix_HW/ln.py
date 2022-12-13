#!/usr/bin/env python
# coding: utf-8


import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='*')
parser.add_argument('-s', '--symbolic', action='store_true')

args = parser.parse_args()

if args.symbolic:
    os.symlink(args.file[0], args.file[1])

