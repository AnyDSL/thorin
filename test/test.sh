for I in *.impala 
do 
    echo $I 
    ./impala -emit-llvm $I 
done
