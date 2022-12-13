#!/usr/bin/env python
# coding: utf-8


import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*')

args = parser.parse_args()

shutil.copy(args.files[0], args.files[1])

