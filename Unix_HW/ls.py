#!/usr/bin/env python
# coding: utf-8

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='?', default=os.getcwd())
parser.add_argument('-a', '--all', action='store_true')

args = parser.parse_args()

if args.all:
    print('.')
    print('..')
    for i in os.listdir(args.directory):
        print(i)
else:
    for i in os.listdir(args.directory):
        if i[0] != '.':
            print(i)
