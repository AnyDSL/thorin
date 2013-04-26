#include <stdio.h>

extern int main_impala(void);
 
int main() {
    printf("main returns: %d\n", main_impala());    
}

