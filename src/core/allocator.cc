#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        bool isEmpty=(freeBlocks.size()==0);
        // =================================== 作业 ===================================
        auto it = freeBlocks.begin();
        for (; it != freeBlocks.end(); it++)
        {
            if (it->second == size)
            {
                break;
            }
            else if (it->second >= size)
            {
                freeBlocks[it->first + size] = it->second - size;
                break;
            }
        }
        size_t addr;
        if (it == freeBlocks.end())
        {   
            if(isEmpty){
                addr=peak;
                peak += size;
            }else{
                it--;
                addr=it->first;
                peak+=(size-it->second);
                freeBlocks.erase(it);
            }

            
        }
        else
        {
            addr=it->first;
            freeBlocks.erase(it);
        }

        usedBlocks[addr]=size;
        used += size;

        // =================================== 作业 ===================================

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
 
        auto it=usedBlocks.find(addr);
        if(it==usedBlocks.end())return;
        usedBlocks.erase(it);
        used-=size;

        freeBlocks[addr]=size;
        it=freeBlocks.find(addr);
        if(it!=freeBlocks.begin()){
            if(std::prev(it)->first+std::prev(it)->second==addr){
                std::prev(it)->second+=it->second;
                it=std::prev(freeBlocks.erase(it));
            }
        }
        if(std::next(it)!=freeBlocks.end()){
            if(it->first+it->second==std::next(it)->first){
                it->second+=std::next(it)->second;
                freeBlocks.erase(std::next(it));
            }
        }



        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
