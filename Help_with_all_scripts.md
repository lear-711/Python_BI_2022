# Description of program features, installation, launch, examples of use:

### Installation:
```
git clone https://github.com/lear-711/Python_BI_2022.git
cd Python_BI_2022
git fetch
git checkout Scripts_bash_HW
cd Unix_HW
```

## 1. **wc.py** 
Displays the number of lines, words, and bytes contained in input file. \
Options: \
     `-c` : The number of bytes in each input file is written to the standard output.  This will cancel out any prior usage of the -m option. \
     `-l` : The number of lines in each input file is written to the standard output. \
     `-w` : The number of words (space-separated) in each input file is written to the standard output.
#### *Program launch:*
`./wc.py <file> <options>`

#### *Examples of use:*

```
$ ./wc.py sample.txt -lwc
	13	423	2859 sample.txt
```
```
$ ./wc.py instruction.txt -c
	2377 instruction.txt
```
You can give 1, 2 or 3 options at once. The first number - number of lines, the second one - number of words, the third one - number of bytes.
     
## 2. **ls.py** 
Displays the contents of directories and information about files \
Option: \
     `-a` : Include directory entries whose names begin with a dot.
     
#### *Program launch:*
`./lc.py <options>`

#### *Examples of use:*
```
$ ./ls.py
tail.py
wc.py
sort.py
instruction.txt
ls.py
cp.py
rm.py
mkdir.py
sample.txt
```
```
$ ./ls.py -a
.
..
tail.py
wc.py
.DS_Store
sort.py
instruction.txt
ls.py
cp.py
rm.py
mkdir.py
.ipynb_checkpoints
sample.txt
```
     
## 3. **sort.py** 
Sorts text and binary files by lines

#### *Program launch:*
`./sort.py <file>`

#### *Examples of use:*
```
$ ./sort.py some.txt 
Ipsum в 60-х годах 
Letraset с образцами Lorem 
Lorem Ipsum - это текст-"рыба"
Lorem Ipsum не только успешно 
Lorem Ipsum является стандартн
PageMaker, в шаблонах 
XVI века. В то время некий безымянный печатник
Его популяризации в
и, в более недавнее время, 
используя Lorem Ipsum для распечатки образцов. 
```
Returns lexicographically sorted lines of given file.

## 4. **rm.py** 
Removes given file(s) \
Option: \
     `-r` : Removing directories and their contents. Recursive deletion.
     
#### *Program launch:*
`./rm.py <options> <file>`

#### *Examples of use:*
Remove instruction.txt file:
```
$ ./rm.py instruction.txt
```
Remove directory my_dir:
```
$ ./rm.py -r my_dir
```
     
## 5. **tail.py** 
Displays last several lines in giver file, default value = 10 lines \
Option: \
     `-n` : number of shown last lines
     
#### *Program launch:*
`./tail.py <options> <file>`

#### *Examples of use:*
```
$ ./tail.py some.txt 
новое время 
послужили публикация
листов 
Letraset с образцами Lorem 
Ipsum в 60-х годах 
и, в более недавнее время, 
программы электронной вёрстки 
типа Aldus 
PageMaker, в шаблонах 
которых используется Lorem Ipsum.
```
```
$ ./tail.py -n 5 some.txt 
и, в более недавнее время, 
программы электронной вёрстки 
типа Aldus 
PageMaker, в шаблонах 
которых используется Lorem Ipsum.
```
     
## 6. **cp.py**
Gives 2 files and copies the contents of the source_file to the target_file \
Option: \
     `-r` : the contents of each named source_file is copied to the destination target_directory
     
#### *Program launch:*
`./cp.py <options> <source_file> <target_file>`

#### *Examples of use:*
```
$ ./cp.py instruction.txt instruction_new.txt 
```
There is new instruction_new.txt file in current directory that is copy of instruction.txt file.
```
$ ./cp.py -r my_dir my_dir2
```
There is new my_dir2 in current directory that is copy of my_dir directory.
     
## 7. **mkdir.py** 
Creates new directory/directories \
Option: \
     `-p` : Create all directories that are specified inside the path. If any directory exists, no warning is displayed.
     
#### *Program launch:*
`./mkdir.py <options> <file>`

#### *Examples of use:*
```
$ ./mkdir.py new_dir
```
```
$ ./mkdir.py -p first_dir/second_dir/last_dir
```
There are new directories: new_dir, first_dir and subfolders: second_dir, last_dir.

## 8. **ln.py**
Creates symbolic link pointing to source named destination \
Option: \
      `-s` : Create a symbolic link

#### *Program launch:*
`./ln.py -s <sourse> <link>`

#### *Examples of use:*
Creates links on files and directories:
```
$ ./ln.py -s instruction.txt instr.txt
```
```
$ ./ln.py -s new_dir new2
```

## 9. **uniq.py**
Removes adjacent duplicate lines from the given text file and creates new file with unique lines.

#### *Program launch:*
`./uniq.py <your_file> <new_file>`

#### *Examples of use:*
```
./uniq.py some.txt unique_some.txt
```
There are only unique lines of some.txt file in unique_some.txt file.
