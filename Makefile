all:
	g++ -lstdc++ -o log main.cpp

log:
	g++ -DLOGGING -lstdc++ -o log main.cpp

.PHONY: all test clean log
