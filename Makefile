all:
	g++ -g -O0 -std=c++11 -pedantic -Wall -lstdc++ -o log main.cpp log.cpp

log:
	g++ -g -O0 -std=c++11 -pedantic -Wall -DLOGGING -lstdc++ -o log main.cpp log.cpp

.PHONY: all test clean log
