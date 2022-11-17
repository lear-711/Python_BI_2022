## Instruction for using the script:

This script works with 2 text files and convert them using regular expressions like this:

1. get all ftp links from text file and write them into new file 'ftps'
2. get and print all numbers from story 2430 A.D.
3. get and print all words that contains letter a/A from story 2430 A.D.
4. get and print all exclamatory sentences from story 2430 A.D.
5. draw barplot of the dependence of the length of unique words on the proportion of such words

Also there are 2 functions:
1. **rus_to_brick** takes *string* that we want to translate to "brick" language and returns translated version of the string
2. **extract_n_words** takes 2 arguments: *string* and *count*. 
It takes *string*, chooses sentences that contain given (*count*) number of words and returns list of tuples each of which contains definite words from found sentences.
