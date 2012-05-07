#include "anydsl/support//symbolmemory.h"

#include <cstdlib>

#include "anydsl/util/assert.h"

namespace anydsl {

const char* SymbolMemory::nullStr="";
const size_t SymbolMemory::bankSize[SymbolMemory::numBanks] = { 16384, 64*2, 256*3, 256*4, 256*5, 256*6 };

SymbolMemory::SymbolMemory() {
    for (size_t i=0; i<numBanks; ++i)
        allocateBank(i);
}

SymbolMemory::~SymbolMemory() {
    size_t s=banks_.size();
    for (size_t i=0; i<s; ++i) {
        free(banks_[i]);
    }
}

void SymbolMemory::allocateBank(size_t bankIdx) {
    Char* bank=(Char*)malloc(sizeof(Char) * bankSize[bankIdx]);
    memoryBank_[bankIdx]=bank;
    memoryIdx_[bankIdx]=0;
    banks_.push_back(bank);
}

SymbolMemory::Char* SymbolMemory::allocateAtBank(size_t bankIdx, size_t size) {
    size_t newPtr=memoryIdx_[bankIdx]+size;
    if (newPtr>bankSize[bankIdx]) {
        allocateBank(bankIdx);
        newPtr=size;
    }
    anydsl_assert(newPtr<bankSize[bankIdx] && memoryBank_[bankIdx]!=0, "Failed to allocate a memory chunk\n")
        << " bank=" << bankIdx
        << " bankSize=" << bankSize[bankIdx]
        << " bankAddr=" << memoryBank_[bankIdx]
        << " allocation size=" << size;
    Char* ret=&memoryBank_[bankIdx][ memoryIdx_[bankIdx] ];
    memoryIdx_[bankIdx]=newPtr;
    return ret;
}

SymbolMemory::Char* SymbolMemory::alloc(size_t length) {
    if (length==0) return (Char*)nullStr;
    if (length<numBanks) return allocateAtBank(length, length+1);
    if (length<marginalSize) return allocateAtBank(0, length+1);
    Char* mem=(Char*)malloc(sizeof(Char)*(length+1));
    banks_.push_back(mem);
    return mem;
}

} // namespace anydsl
