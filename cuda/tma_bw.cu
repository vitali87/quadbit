// Probe the L2->smem TMA bandwidth ceiling at varying occupancy. Source is a 32MB buffer
// (fits the 128MB L2, so loads hit L2 not DRAM). Each CTA loops issuing 1D cp.async.bulk
// loads of CHUNK bytes into a small double-buffered smem (small => many blocks/SM => high
// occupancy). If aggregate GB/s >> the matmul's 3.96 TB/s, the matmul is occupancy/issue
// limited (1 block/SM), not L2-BW limited => a higher-occupancy kernel could go faster.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define CHUNK 16384      // 16KB per bulk load
#define NBUF 2
#define SMEM (NBUF * CHUNK + 64)

__global__ void __launch_bounds__(128) bw(const uint8_t *src, size_t srcbytes, int iters, uint32_t *sink) {
    extern __shared__ __align__(128) uint8_t sm[];
    uint64_t *bar = (uint64_t *)(sm + NBUF * CHUNK);
    int tid = threadIdx.x;
    if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"((uint32_t)__cvta_generic_to_shared(&bar[0]))); asm volatile("fence.proxy.async.shared::cta;"); }
    __syncthreads();
    size_t off = ((size_t)blockIdx.x * 1024) % (srcbytes - CHUNK);   // spread starts across L2-resident buf
    uint32_t acc = 0;
    for (int i = 0; i < iters; i++) {
        uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[0]);
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(b),"r"((uint32_t)CHUNK));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::
                         "r"((uint32_t)__cvta_generic_to_shared(&sm[(i&1)*CHUNK])), "l"(src+off), "r"((uint32_t)CHUNK), "r"(b));
        }
        uint32_t par = i & 1;
        asm volatile("{\n\t.reg .pred p;\nW:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra W;\n\t}\n"::"r"(b),"r"(par));
        acc += sm[(i&1)*CHUNK + tid];
        off += CHUNK; if (off > srcbytes - CHUNK) off = 0;
    }
    if (tid == 999) sink[blockIdx.x] = acc;
}

int main() {
    size_t srcbytes = 32ull << 20;   // 32MB, L2-resident
    uint8_t *src; uint32_t *sink;
    cudaMalloc(&src, srcbytes); cudaMemset(src, 1, srcbytes);
    cudaFuncSetAttribute(bw, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    for (int blocks : {188, 376, 752, 1504, 3008}) {
        int iters = 2000;
        cudaMalloc(&sink, blocks * 4);
        for (int w = 0; w < 3; w++) bw<<<blocks, 128, SMEM>>>(src, srcbytes, iters, sink);
        cudaDeviceSynchronize();
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s); bw<<<blocks, 128, SMEM>>>(src, srcbytes, iters, sink);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e);
        double gb = (double)blocks * iters * CHUNK;
        if (cudaGetLastError() != cudaSuccess) { printf("err\n"); }
        printf("tma_bw blocks=%5d: %.3f ms  %.1f GB/s  (%.2f TB/s)\n", blocks, ms, gb/(ms/1e3)/1e9, gb/(ms/1e3)/1e12);
        cudaFree(sink);
    }
    return 0;
}
