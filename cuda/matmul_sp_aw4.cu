// Asymmetric-A wide kernel: A on WK=4 (128B box -> 128B swizzle, the most efficient mode)
// while B stays WK=2 (128B box, 128B swizzle). A super-chunk = 4 k128, B chunk = 2 k128,
// so A is loaded half as often and on the wider/more-efficient swizzle. Separate A/B
// pipelines (STAGES_A=2, STAGES_B=2) since their cadences differ; an A super-chunk spans
// 2 B-chunks. Goal: push the unit kernel's 6.0 TB/s swizzle floor toward 7.3. Shared-B
// 256x128, bf16 out, all-ones => out == Klog/2 (perf check; real-data verify follows).
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 128
#define BN 128
#define AROWB 32             // bytes per k128 of compressed A
#define BROWB 64             // bytes per k128 of full B
#define AWB 128              // A super-chunk row bytes (4 k128 x 32)
#define BWB 128              // B chunk row bytes (2 k128 x 64)
#define SA 2                 // A pipeline stages
#define SB 2                 // B pipeline stages
#define ASZ (BM * AWB)       // 16384 per A-half
#define BSZ (BN * BWB)       // 16384 B chunk
#define SMEM (2 * SA * ASZ + SB * BSZ + (2 * SA + 2 * SB) * 8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          __nv_bfloat16 *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem + wg * SA * ASZ;          // this WG's A buffers
    uint8_t *b_s = smem + 2 * SA * ASZ;           // shared B buffers
    uint64_t *fullA = (uint64_t *)(b_s + SB * BSZ);
    uint64_t *emptyA = fullA + SA;
    uint64_t *fullB = emptyA + SA;
    uint64_t *emptyB = fullB + SB;

    int block_row = blockIdx.y * (2 * BM) + wg * BM;
    int a_load_row = blockIdx.y * (2 * BM), block_col = blockIdx.x * BN;
    int chunksB = Klog / 256;   // each B chunk = 2 k128
    int chunksA = Klog / 512;   // each A super-chunk = 4 k128

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x38383838u, sb = 0x38383838u, meta = 0x44444444u;
    uint16_t z = 0;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < SA; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&fullA[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"((uint32_t)__cvta_generic_to_shared(&emptyA[s])));
        }
#pragma unroll
        for (int s = 0; s < SB; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&fullB[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"((uint32_t)__cvta_generic_to_shared(&emptyB[s])));
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    auto issueA = [&](int s, int a) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&fullA[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * ASZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])), "l"(&mapA), "r"(a * AWB), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[SA * ASZ + s * ASZ])), "l"(&mapA), "r"(a * AWB), "r"(a_load_row + BM), "r"(bar));
    };
    auto issueB = [&](int s, int cb) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&fullB[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)BSZ));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(cb * BWB), "r"(block_col), "r"(bar));
    };
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < SA; s++) if (s < chunksA) issueA(s, s);
#pragma unroll
        for (int s = 0; s < SB; s++) if (s < chunksB) issueB(s, s);
    }

    for (int cb = 0; cb < chunksB; cb++) {
        int a = cb >> 1, asub = cb & 1;
        int sB = cb % SB; uint32_t parB = (cb / SB) & 1;
        int sA = a % SA; uint32_t parA = (a / SA) & 1;
        asm volatile("{\n\t.reg .pred p;\nWB:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WB;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&fullB[sB])), "r"(parB));
        if (asub == 0)
            asm volatile("{\n\t.reg .pred p;\nWAA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WAA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&fullA[sA])), "r"(parA));

        int aoff = sA * ASZ, boff = sB * BSZ;
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int kA = asub * 2 + sub;   // A k128 within the 4-k128 super-chunk
            uint32_t af[4][4], bf[8][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * AWB + kA * AROWB + acblk * 16; ao ^= ((ao >> 7) & 7) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BWB + sub * BROWB + bsub * 16; bo ^= ((bo >> 7) & 7) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                          "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&emptyB[sB])));
        if (asub == 1)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&emptyA[sA])));
        if (tid == 0) {
            int nB = cb + SB;
            if (nB < chunksB) {
                asm volatile("{\n\t.reg .pred p;\nWEB:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WEB;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&emptyB[sB])), "r"(parB));
                issueB(sB, nB);
            }
            if (asub == 1) {
                int nA = a + SA;
                if (nA < chunksA) {
                    asm volatile("{\n\t.reg .pred p;\nWEA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WEA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&emptyA[sA])), "r"(parA));
                    issueA(sA, nA);
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + b_col[n] + (lane & 3) * 2;
            *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0], acc[idx][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2], acc[idx][3]);
        }
}
static void mk(CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo) {
    uint64_t gd[2]={(uint64_t)inner,(uint64_t)outer}; uint64_t gs[1]={(uint64_t)inner}; uint32_t bd[2]={(uint32_t)bi,(uint32_t)bo},es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,p,gd,gs,bd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("map fail %s\n",s);}
}
void run(int sz){
    int M=sz,N=sz,Klog=sz,KAb=Klog/4,KBb=Klog/2; uint8_t*dA,*dB; __nv_bfloat16*dC;
    cudaMalloc(&dA,(size_t)M*KAb); cudaMalloc(&dB,(size_t)N*KBb); cudaMalloc(&dC,(size_t)M*N*2);
    cudaMemset(dA,0x22,(size_t)M*KAb); cudaMemset(dB,0x22,(size_t)N*KBb);
    alignas(64) CUtensorMap mapA,mapB; mk(&mapA,dA,KAb,M,AWB,BM); mk(&mapB,dB,KBb,N,BWB,BN);
    cudaFuncSetAttribute(matmul_sp,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    dim3 grid(N/BN,M/(2*BM)),block(256);
    for(int i=0;i<5;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Klog);
    if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);int it=20;cudaEventRecord(s);
    for(int i=0;i<it;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Klog);
    cudaEventRecord(e);cudaEventSynchronize(e);float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=it;
    double gf=2.0*M*N*Klog/(ms/1e3)/1e9;
    __nv_bfloat16*hC=(__nv_bfloat16*)malloc((size_t)M*N*2);cudaMemcpy(hC,dC,(size_t)M*N*2,cudaMemcpyDeviceToHost);
    int wrong=0; for(size_t i=0;i<(size_t)M*N;i++) if(__bfloat162float(hC[i])!=(float)(Klog/2)) wrong++;
    printf("matmul_sp_aw4 (A WK=4 128B-swz, B WK=2): %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (out0=%.0f exp %d)\n",
           M,N,Klog,ms,gf,wrong==0?"PASS":"FAIL",__bfloat162float(hC[0]),Klog/2);
    free(hC);cudaFree(dA);cudaFree(dB);cudaFree(dC);
}
int main(){ run(2048); run(4096); run(8192); return 0; }
