#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs='*')
parser.add_argument('-l', '--lines', action='store_true')
parser.add_argument('-w', '--words', action='store_true')
parser.add_argument('-c', '--bytes', action='store_true')

args = parser.parse_args()

size = 0

if len(args.file) == 0:
    lines = 0
    words = 0
    path = ""
    for line in sys.stdin:
        lines += 1
        path += line
        line = line.split()
        for word in line:
            words += 1
    size = len(path.encode('utf-8'))
else:
    with open(args.file[0], 'r') as file:
        lines = len(file.readlines())
        file.seek(0)
        read_file = file.read()
        words_in_file = read_file.split()
        words = len(words_in_file)
        size = os.path.getsize(args.file[0])


result = ''

if args.lines:
    result += "\t" + str(lines)
if args.words:
    result += "\t" + str(words)
if args.bytes:
    result += "\t" + str(size)
if (not args.lines) and (not args.words) and (not args.bytes):
    result += "\t" + str(lines) + "\t" + str(words) + "\t" + str(size)

if len(args.file) != 0:
    result += " " + args.file[0]

print(result)
