#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini
{

    Shape infer_broadcast(const Shape &A, const Shape &B)
    {
        int rankA = A.size();
        int rankB = B.size();
        int rankNew = std::max(rankA, rankB);

        Shape shape(rankNew, 0);
        for (int i = 0; i < rankNew; i++)
        {
            int da = rankA - i - 1 < 0 ? 1 : A[rankA - i - 1];
            int db = rankB - i - 1 < 0 ? 1 : B[rankB - i - 1];
            int d = 0;
            if (da == db)
            {
                d = da; // 两个相等 (比如 5 和 5)，取 5
            }
            else if (da == 1)
            {
                d = db; // A 是 1，广播 B
            }
            else if (db == 1)
            {
                d = da; // B 是 1，广播 A
            }
            else
            {
                // 既不相等，也没人是 1 (比如 3 和 5)，无法广播
                // 这种情况下通常返回空 shape 表示失败，或者抛异常
                return {};
            }

            shape[rankNew - i - 1] = d;
        }
        // =================================== 作业 ===================================
        // =================================== 作业 ===================================

        return shape;
    }

    int get_real_axis(const int &axis, const int &rank)
    {
        IT_ASSERT(rank >= 1);
        IT_ASSERT(axis >= -rank && axis <= (rank - 1));
        int newAxis;
        if (axis < 0)
        {
            newAxis = rank + axis;
        }
        else
        {
            newAxis = axis;
        }
        return newAxis;
    }

    Shape locate_index(size_t inputN, const Shape &shape)
    {
        Shape ans(shape.size());
        auto i = ans.rbegin();
        auto j = shape.rbegin(), ej = shape.rend();
        while (j != ej)
        {
            auto div = std::div(inputN, *j++);
            *i++ = div.rem;
            inputN = div.quot;
        }
        return ans;
    }

    size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                          const Shape &stride)
    {
        size_t ans = 0;
        Shape index(shapeIndex.size());
        IT_ASSERT(shapeIndex.size() == shape.size());
        IT_ASSERT(shape.size() == stride.size());
        for (size_t i = 0; i < shape.size(); ++i)
        {
            index[i] = shapeIndex[i] % shape[i];
            ans += index[i] * stride[i];
        }
        return ans;
    }

    std::string device_to_str(Device device)
    {
        std::string deviceStr;
        switch (device)
        {
        case Device::CPU:
            return "CPU";
        default:
            IT_TODO_HALT();
        }
    }

    std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs)
    {
        std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
        std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
        return deviceStr + ", " + opStr;
    }

} // namespace infini
