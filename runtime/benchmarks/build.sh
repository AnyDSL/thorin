gcc support.c -c -O2 -mavx -Wall
./impala extern.impala mapping_cpu.impala jacobi.impala -emit-llvm
opt -O3 jacobi.bc -o jacobi_opt.bc
llc jacobi_opt.bc -mattr=avx
gcc jacobi_opt.s support.o
