#ifndef ANYDSL_SYMBOL_MEMORY_HEADER
#define ANYDSL_SYMBOL_MEMORY_HEADER

#include <cstdlib>
#include <vector>

namespace anydsl {

class SymbolMemory {
public:
    typedef char Char;
    static const size_t numBanks = 6;
    static const size_t bankSize[numBanks];
    static const size_t marginalSize = 256;

    static const char* nullStr;

    SymbolMemory();
    ~SymbolMemory();
    Char* alloc(size_t length);
private:
    void allocateBank(size_t bankIdx);
    Char* allocateAtBank(size_t bankIdx, size_t size);
    Char* memoryBank_[numBanks];
    size_t memoryIdx_[numBanks];
    std::vector<Char*> banks_;
};

} // namespace anydsl

#endif
