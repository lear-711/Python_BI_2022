#!/usr/bin/env python
# coding: utf-8

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='*')
parser.add_argument('-p', '--parents', action='store_true')

args = parser.parse_args()

parent_dir = "./"
path = os.path.join(parent_dir, args.directory[0])

if args.parents:
    os.makedirs(path, exist_ok=True)
else:
    os.mkdir(path, mode=0o777)

