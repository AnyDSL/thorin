gcc support.c -c -O2 -mavx -Wall
./impala extern.impala mapping_cpu.impala gaussian.impala -emit-llvm
opt -O3 gaussian.bc -o gaussian_opt.bc
llc gaussian_opt.bc -mattr=avx
gcc gaussian_opt.s support.o
