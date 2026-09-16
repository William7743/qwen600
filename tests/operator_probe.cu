// Isolated calls to production kernels. Fixtures are float32, quantized to BF16 on input.
#include "models/qwen_model.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

struct Buffers {
    std::vector<void*> owned;
    ~Buffers() { for (void* p : owned) cudaFree(p); }
    template<class T> T* alloc(size_t n) {
        T* p; CUDA_CHECK(cudaMalloc(&p,n*sizeof(T))); owned.push_back(p); return p;
    }
};

int main(int argc, char** argv) {
    try {
        if(argc!=8) throw std::runtime_error("operator_probe OP INPUT OUTPUT A B C VERSION");
        std::string op=argv[1]; int a=std::stoi(argv[4]), b=std::stoi(argv[5]), c=std::stoi(argv[6]);
        // argv[7] is a format version so stale binaries cannot silently change layouts.
        if(std::string(argv[7])!="1") throw std::runtime_error("Invalid fixture version");
        std::ifstream input(argv[2],std::ios::binary);
        std::ofstream output(argv[3],std::ios::binary);
        if(!input || !output) throw std::runtime_error("Cannot open fixture/output");
        Buffers mem;
        auto read=[&](size_t n) {
            std::vector<float> v(n); input.read(reinterpret_cast<char*>(v.data()),n*sizeof(float));
            if(!input) throw std::runtime_error("Truncated fixture"); return v;
        };
        auto bf=[&](size_t n) {
            auto v=read(n); std::vector<bf16> h(n);
            for(size_t i=0;i<n;++i) h[i]=__float2bfloat16_rn(v[i]);
            bf16* p=mem.alloc<bf16>(n); CUDA_CHECK(cudaMemcpy(p,h.data(),n*sizeof(bf16),cudaMemcpyHostToDevice)); return p;
        };
        auto emit=[&](float* p,size_t n) {
            std::vector<float> v(n); CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(float),cudaMemcpyDeviceToHost));
            output.write(reinterpret_cast<char*>(v.data()),n*sizeof(float));
        };
        auto emit_bf=[&](bf16* p,size_t n) {
            std::vector<bf16> v(n); CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(bf16),cudaMemcpyDeviceToHost));
            for(auto x:v) { float f=__bfloat162float(x); output.write(reinterpret_cast<char*>(&f),sizeof(f)); }
        };
        if(op=="attention") {
            if(a<1 || a>SEQ_LEN) throw std::runtime_error("Bad context length");
            RunState s{}; s.q=bf(Q_DIM);
            s.key_cache=bf(size_t(SEQ_LEN)*KV_DIM); s.value_cache=bf(size_t(SEQ_LEN)*KV_DIM);
            s.att=mem.alloc<float>(N_HEADS*SEQ_LEN);
            std::vector<float> sentinel(N_HEADS*SEQ_LEN,-1234.f);
            CUDA_CHECK(cudaMemcpy(s.att,sentinel.data(),sentinel.size()*sizeof(float),cudaMemcpyHostToDevice));
            auto original_q=mem.alloc<bf16>(Q_DIM);
            CUDA_CHECK(cudaMemcpy(original_q,s.q,Q_DIM*sizeof(bf16),cudaMemcpyDeviceToDevice));
            attention_qk_kernel<<<N_HEADS,std::min(1024,a)>>>(s.att,s.q,s.key_cache,a-1);
            emit(s.att,N_HEADS*SEQ_LEN);
            softmax_kernel<<<N_HEADS,1>>>(s.att,a-1); emit(s.att,N_HEADS*SEQ_LEN);
            attention_v_kernel<<<N_HEADS,HEAD_DIM>>>(s.q,s.att,s.value_cache,a-1); emit_bf(s.q,Q_DIM);
            // Also exercise the dispatcher, not just the constituent kernels.
            CUDA_CHECK(cudaMemcpy(s.q,original_q,Q_DIM*sizeof(bf16),cudaMemcpyDeviceToDevice));
            attention_gpu(&s,0,a-1); emit_bf(s.q,Q_DIM);
        } else if(op=="softmax") {
            if(a<1 || a>SEQ_LEN) throw std::runtime_error("Bad softmax length");
            auto v=read(N_HEADS*SEQ_LEN); auto p=mem.alloc<float>(v.size());
            CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));
            softmax_kernel<<<N_HEADS,1>>>(p,a-1); emit(p,v.size());
        } else if(op=="rmsnorm") {
            auto x=bf(DIM), w=bf(DIM), y=c ? x : mem.alloc<bf16>(DIM);
            rmsnorm_gpu(y,x,w,DIM); emit_bf(y,DIM);
        } else if(op=="qknorm") {
            auto q=bf(Q_DIM), k=bf(KV_DIM), qw=bf(HEAD_DIM), kw=bf(HEAD_DIM);
            qk_norm_fused_gpu(q,k,qw,kw); emit_bf(q,Q_DIM); emit_bf(k,KV_DIM);
        } else if(op=="rope") {
            if(a<0 || a>=SEQ_LEN) throw std::runtime_error("Bad RoPE position");
            auto q=bf(Q_DIM), k=bf(KV_DIM); rope_gpu_naive(q,k,a);
            emit_bf(q,Q_DIM); emit_bf(k,KV_DIM);
        } else if(op=="swiglu") {
            auto gate=bf(HIDDEN_DIM), up=bf(HIDDEN_DIM);
            swiglu_gpu(gate,up,HIDDEN_DIM); emit_bf(gate,HIDDEN_DIM);
        } else if(op=="convert") {
            auto x=bf(VOCAB_SIZE); auto y=mem.alloc<float>(VOCAB_SIZE);
            convert_bf16_to_fp32_kernel<<<(VOCAB_SIZE+255)/256,256>>>(x,y,VOCAB_SIZE); emit(y,VOCAB_SIZE);
        } else if(op=="matmul") {
            if(a<1 || a>VOCAB_SIZE || b<1 || b>HIDDEN_DIM || (c!=0 && c!=1)) throw std::runtime_error("Bad matmul shape");
            auto w=bf(size_t(a)*b), x=bf(b), y=bf(a);
            cublasHandle_t handle; if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS init failed");
            matmul_cublas(handle,y,w,x,a,b,1.f,float(c)); emit_bf(y,a); cublasDestroy(handle);
        } else throw std::runtime_error("Unknown operator");
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        if(input.peek()!=EOF) throw std::runtime_error("Extra fixture data");
        if(!output) throw std::runtime_error("Output write failed");
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 2; }
}
