
# ### Функция для прочтения файла по 4 строки:

def read_4lines(file):
    lines = []
    for i in range(4):
        line = file.readline()
        if line == "":
            return []
        lines.append(line.replace("\n", ""))
    return lines



# ###  Функция для проверки GC-состава:

def gc_bounds_func(inter_gc, lines):
    cnt_gc = 0
    one_read = lines[1]
    gc_count = (one_read.count('C') + one_read.count('G'))/len(one_read)*100

    if type(inter_gc) == int:
        if gc_count < inter_gc:
            cnt_gc =1
                
    else:
        min_gc = inter_gc[0]
        max_gc = inter_gc[1]
        if (gc_count > min_gc) and (gc_count < max_gc):
            cnt_gc = 1
    return cnt_gc




# ### Функция для проверки длины прочтения:

def length_bounds_func(length_bounds, lines):
    cnt_len = 0
    len_read = len(lines[1])

    if type(length_bounds) == int:
        if len_read < length_bounds:
            cnt_len = 1
    else:
        min_len = length_bounds[0]
        max_len = length_bounds[1]
        if (len_read > min_len) and (len_read < max_len):
            cnt_len = 1
    return cnt_len



# ### Функция для оценки качества прочтения:

def quality_threshold_func(quality_threshold, lines):
    
    qual_line = lines[3]
    cnt_qual = 0
    quality_list = []
    for each in list(qual_line):
        quality_list.append(ord(each))

    median_quality = sum(quality_list) / len(quality_list)
    if median_quality > quality_threshold:
        cnt_qual = 1
    return cnt_qual


# ### Основная функция, фильтрующая риды и записывающая риды в файл:


def main(input_fastq, output_file_prefix, gc_bounds = (0, 100), length_bounds = (0, 2**32), 
         quality_threshold = 0, save_filtered = False):
    
    passed_file = output_file_prefix + "_passed.fastq"
    failed_file = output_file_prefix + "_failed.fastq"
    
    with open(input_fastq) as file:
        while True:
            lines = read_4lines(file)
            if len(lines) == 0:
                break
            cnt_gc = gc_bounds_func(gc_bounds, lines)
            cnt_len = length_bounds_func(length_bounds, lines)
            cnt_qual = quality_threshold_func(quality_threshold, lines)

            if (cnt_gc + cnt_len + cnt_qual) == 3:
                with open(passed_file, "a") as file_2:
                    file_2.write("\n".join(lines))
            else:
                if save_filtered == True:
                    with open(failed_file, "a") as file_3:
                        file_3.write("\n".join(lines))




