# Manual for correct use functions in Functional.py script

Functions:
1. **sequential_map** takes several positional arguments (functions and the last one - container with values) and returns list of results of successively applying passed functions to values in the container;
2. **consensus_filter** takes as arguments any number of functions that return True or False and a container with some values; returns a list of values that, when passed to all functions, evaluate to True;
3. **conditional_reduce** takes 2 functions and a container with some values (the first function must take 1 argument and return True or False, the second also takes 2 arguments and returns a value); \
returns one value is the result of reduce, skipping the values with which the first function returned False;
5. **func_chain** tajes any number of functions as arguments; returns a function concatenating all passed by sequential execution;
6. **multiple_partial** - analogue of the partial function, but which takes an unlimited number of functions as arguments and returns a list of the same number of "partial functions";
7. **my_print** - full analogue of the print function.
