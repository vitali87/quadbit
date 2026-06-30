// Test: does loading W k128-slices per TMA (wider contiguous inner) raise the mem-only
// L2->smem bandwidth toward the 7.3 TB/s ceiling (vs 3.96 TB/s with narrow per-step boxes)?
// Shared-B 256x128, mem-only (no compute). A box {AROWB*W,256}, B box {BROWB*W,128}.
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#ifndef WK
#define WK 4
#endif
#define BM 128
#define BN 128
#define AROWB 32
#define BROWB 64
#define STAGES 3
#define ASZ (BM * AROWB * WK)
#define BSZ (BN * BROWB * WK)
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + STAGES * 8 + 128)

__global__ void __launch_bounds__(256)
mw(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB, float *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);
    int a_load_row = blockIdx.y * (2 * BM), block_col = blockIdx.x * BN;
    int ksteps = Klog / (128 * WK);   // wide steps
    float acc = 0.f;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();
    auto issue = [&](int s, int step) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(bar),"r"((uint32_t)(2*ASZ+BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[s*ASZ])),"l"(&mapA),"r"(step*AROWB*WK),"r"(a_load_row),"r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES*ASZ+s*ASZ])),"l"(&mapA),"r"(step*AROWB*WK),"r"(a_load_row+BM),"r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s*BSZ])),"l"(&mapB),"r"(step*BROWB*WK),"r"(block_col),"r"(bar));
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++) if (s < ksteps) issue(s, s);
    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES; uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nW:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra W;\n\t}\n"::"r"((uint32_t)__cvta_generic_to_shared(&full[s])),"r"(par));
        acc += (float)a_s[s*ASZ + wtid] + (float)b_s[s*BSZ + wtid];
        asm volatile("bar.sync 1;");
        int next = step + STAGES; if (tid == 0 && next < ksteps) issue(s, next);
    }
    if (tid == 999) C[0] = acc;
}
static void mk(CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo) {
    uint64_t gd[2]={(uint64_t)inner,(uint64_t)outer}; uint64_t gs[1]={(uint64_t)inner}; uint32_t bd[2]={(uint32_t)bi,(uint32_t)bo},es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,p,gd,gs,bd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("map fail %s\n",s);}
}
void run(int sz){
    int M=sz,N=sz,Klog=sz,KAb=Klog/4,KBb=Klog/2; uint8_t*dA,*dB; float*dC;
    cudaMalloc(&dA,(size_t)M*KAb); cudaMalloc(&dB,(size_t)N*KBb); cudaMalloc(&dC,256);
    cudaMemset(dA,0x22,(size_t)M*KAb); cudaMemset(dB,0x22,(size_t)N*KBb);
    alignas(64) CUtensorMap mapA,mapB; mk(&mapA,dA,KAb,M,AROWB*WK,BM); mk(&mapB,dB,KBb,N,BROWB*WK,BN);
    cudaFuncSetAttribute(mw,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    dim3 grid(N/BN,M/(2*BM)),block(256);
    for(int i=0;i<5;i++) mw<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Klog);
    if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}
    cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e); int it=20; cudaEventRecord(s);
    for(int i=0;i<it;i++) mw<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Klog);
    cudaEventRecord(e);cudaEventSynchronize(e); float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=it;
    double bytes=(double)( (size_t)M*KAb*(N/BN) + (size_t)N*KBb*(M/(2*BM)) ); // A loaded N/BN x, B loaded M/256 x
    printf("memwide WK=%d %dx%d: %.3f ms  %.2f TB/s  (eqv %.0f GFLOP/s)\n",WK,sz,sz,ms,bytes/(ms/1e3)/1e12, 2.0*M*N*Klog/(ms/1e3)/1e9);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
}
int main(){ run(8192); return 0; }
