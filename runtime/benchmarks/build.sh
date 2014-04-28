gcc support.c -c -O2 -mavx -Wall
./impala extern.impala mapping_cpu.impala $1 -emit-llvm
base=${1%.*}
opt -O3 ${base}.bc -o ${base}_opt.bc
llc ${base}_opt.bc -mattr=avx
gcc ${base}_opt.s support.o
