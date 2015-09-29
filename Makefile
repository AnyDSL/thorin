all:
	g++ -std=c++11 -pedantic -Wall -lstdc++ -o log main.cpp

log:
	g++ -std=c++11 -pedantic -Wall -DLOGGING -lstdc++ -o log main.cpp

.PHONY: all test clean log
