#ifndef PRINTABLE_H
#define PRINTABLE_H

#include <ostream>

class Printable {
	public:
		virtual const void print(std::ostream &out) const = 0;
};

#endif
