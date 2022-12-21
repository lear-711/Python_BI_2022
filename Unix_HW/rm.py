#!/usr/bin/env python
# coding: utf-8


import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='*')
parser.add_argument('-r', '--recursive', action='store_true')

args = parser.parse_args()

if args.recursive:
    for file in args.file:
        shutil.rmtree(file, ignore_errors=True)
else:
    for file in args.file:
        os.remove(file)

