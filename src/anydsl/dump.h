#ifndef ANYDSL_DUMP_H
#define ANYDSL_DUMP_H

#include <iostream>

namespace anydsl {

class Printer;

enum LambdaPrinterMode {
	LAMBDA_PRINTER_MODE_DEFAULT,
	LAMBDA_PRINTER_MODE_SKIPBODY,
};

} // namespace anydsl

#endif
