all:
	gcc -o log main.cpp

log:
	gcc -DLOGGING -o log main.cpp
