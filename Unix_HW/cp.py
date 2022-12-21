#!/usr/bin/env python
# coding: utf-8


import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*')
parser.add_argument('-r', '--recursive', action='store_true')

args = parser.parse_args()

if args.recursive:
    shutil.copytree(args.files[0], args.files[1])
else:
    shutil.copy(args.files[0], args.files[1])

