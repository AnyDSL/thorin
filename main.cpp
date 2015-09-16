#include <stdio.h>
#include <stdlib.h>

#include "log.h"

LOG_LEVEL Logging::level = INFO;

int main(int argc, const char* argv[]) {
  if(argc < 3) {
    exit(-1);
  }

  if(atoi(argv[1]) == 0) {
    Logging::level = DEBUG;
  }

  int runs = atoi(argv[2]);

  LOG(INFO, "Starting %i runs!\n", runs);
  for(int i = 0; i < runs; i++) {
    LOG(DEBUG, "-> %i / %i\n", i, runs);
  }
  LOG(INFO, "Finished!\n");
}
