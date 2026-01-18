#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        auto Ashape = inputs[0]->getDims();
        auto Bshape = inputs[1]->getDims();


        int m_A=Ashape[Ashape.size()-2];
        int k_A=Ashape[Ashape.size()-1];
        if(transA){std::swap(m_A,k_A);};

        int k_B=Bshape[Bshape.size()-2];
        int n_B=Bshape[Bshape.size()-1];
        if(transB){std::swap(k_B,n_B);};
        
        if(k_A!=k_B){return {};}


        //1 2 3
        //  2 3
        Shape s;
        int rank=std::max(Ashape.size(),Bshape.size());
        for(int i=rank;i>2;i--){
            int a=Ashape.size()-i<0?1:Ashape[Ashape.size()-i];
            int b=Bshape.size()-i<0?1:Bshape[Bshape.size()-i];

            int d;
            if(a==1||a==b){
                d=b;
            }else if(b==1){
                d=a;
            }else{
                return {};
            }
            s.push_back(d);
        }     
        s.push_back(m_A);
        s.push_back(n_B);

        return {{s}};


        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
    }

} // namespace infini