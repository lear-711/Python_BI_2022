
transcribe_dict = {'A': 'U', 'a':'u', 'G':'C', 'g':'c', 'T':'A', 't':'a', 'C':'G', 'c':'g'}
complement_DNA_dict = {'A': 'T', 'a':'t', 'G':'C', 'g':'c', 'T':'A', 't':'a', 'C':'G', 'c':'g'}
complement_RNA_dict = {'A': 'U', 'a':'u', 'G':'C', 'g':'c', 'U':'A', 'u':'a', 'C':'G', 'c':'g'}

def sort_set_upper_nucl_seq(nucl_seq):
    a = list(nucl_seq)
    b = []
    for each in a:
        b.append(each.upper())
    return sorted(set(b))


comm = input('Введите команду: ')

while comm != 'exit':
    nucl_seq = input('Введите последовательность нуклеиновой кислоты: ')
    list_nucl_seq = sort_set_upper_nucl_seq(nucl_seq)
    if (list_nucl_seq <= ['A', 'C', 'G', 'T'] or list_nucl_seq <= ['A', 'C', 'G', 'U']):
        if comm == 'transcribe':
            if list_nucl_seq == ['A', 'C', 'G', 'T']:
                res_list = []
                for each in list(nucl_seq):
                    res_list.append(transcribe_dict.get(each))
                print(''.join(res_list))
            else:
                print("Вы ввели последовательность РНК! Уточните команду и введите корректную последовательность!")

  # [transcribe_dict[nucl] for nucl in nucl_seq]

        elif comm == 'reverse':
            print(nucl_seq[::-1])
        
        elif comm == 'complement':
            if list_nucl_seq == ['A', 'C', 'G', 'T']:
                res_list = []
                for each in list(nucl_seq):
                    res_list.append(complement_DNA_dict.get(each))
                print(''.join(res_list))
            else:
                res_list = []
                for each in list(nucl_seq):
                    res_list.append(complement_RNA_dict.get(each))
                print(''.join(res_list))
                      
        elif comm == 'reverse complement':   
            if list_nucl_seq == ['A', 'C', 'G', 'T']:
                res_list = []
                for each in list(nucl_seq):
                    res_list.append(complement_DNA_dict.get(each))
                print(''.join(res_list)[::-1])
            else:
                res_list = []
                for each in list(nucl_seq):
                    res_list.append(complement_RNA_dict.get(each))
                print(''.join(res_list)[::-1])
    else:
        print("Введенная последовательность не является ДНК или РНК, попробуйте еще раз!")
    
    comm = input('Введите команду: ')
    
print("Удачи!")




