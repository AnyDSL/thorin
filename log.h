enum LOG_LEVEL {
  DEBUG, INFO
};

#define LOG(level, format, ...) { \
  printf(format, __VA_ARGS__);        \
}
