#include <stdio.h>

#include "log.h"

int main(int argc, const char* argv[]) {
  if(argc < 2) {
    exit(-1);
  }

  int runs = atoi(argv[1]);

  LOG(LOG_LEVEL::INFO, "Starting %i runs!", runs);
  for(int i = 0; i < runs; i++) {
    LOG(LOG_LEVEL::DEBUG, "-> %i / %i", i, runs);
  }
  LOG(LOG_LEVEL::INFO, "Finished!");
}
