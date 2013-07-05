#!/bin/bash
# parameter 1: Impala program with a 'foo' function that returns an int
rm -f a.out
# generate LLVM code
anydsl2 -i $1 --emit-llvm -O >& out.ll
# generate assembler
llc out.ll
# link with a C main function
cc execute_impala.c out.s
echo "Execute Impala program:" $1
./a.out
# clean up
rm -f out.ll
rm -f out.s
