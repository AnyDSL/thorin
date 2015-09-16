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
  for(Counter c; c.get() < runs; c.inc()) {
    LOG(DEBUG, "-> %i / %i\n", c.get(), runs);
  }
  LOG(INFO, "Finished!\n");
}
