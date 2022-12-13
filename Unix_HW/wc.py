#!/usr/bin/env python
# coding: utf-8

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument('-l', '--lines', action='store_true')
parser.add_argument('-w', '--words', action='store_true')
parser.add_argument('-c', '--bytes', action='store_true')

args = parser.parse_args()


with open(args.file, 'r') as file:
    lines = len(file.readlines())
    file.seek(0)
    read_file = file.read()
    words = read_file.split()

result = ''

if args.lines:
    result += "\t" + str(lines)
if args.words:
    result += "\t" + str(len(words))
if args.bytes:
    result += "\t" + str(os.path.getsize(args.file))

result += " " + args.file

print(result)
