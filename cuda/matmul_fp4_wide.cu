// Dense FP4 with the sparse breakthrough's structure: shared-B 256x128 + WK=4 wide
// 128B-swizzled TMA (4 k64 slices/load) on mma.sync m16n8k64. Dense was load-limited
// (mem-only 0.630ms vs full 0.731) at 64B boxes; wide loads should help. bf16 out.
// all-ones => out == Klog (dense, no sparsity). ue8m0 unit scales (0x7f), 2X.
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 128
#define BN 128
#define WK 4                 // k64 slices per TMA load
#define KB64 32              // bytes per k64 slice (64 fp4)
#define AW (KB64 * WK)       // 128
#define BWB (KB64 * WK)      // 128
#define STAGES 2
#define ASZ (BM * AW)        // 16384 per A-half
#define BSZ (BN * BWB)       // 16384 shared B
#define SMEM (2*STAGES*ASZ + STAGES*BSZ + 2*STAGES*8 + 128)

__global__ void __launch_bounds__(256)
matmul_fp4(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
           __nv_bfloat16 *C, int M, int N, int Kfp4) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);
    uint64_t *empty = full + STAGES;
    int block_row = blockIdx.y * (2 * BM) + wg * BM;
    int a_load_row = blockIdx.y * (2 * BM), block_col = blockIdx.x * BN;
    int chunks = Kfp4 / (64 * WK);

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
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
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();
    auto issue = [&](int s, int chunk) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(chunk * BWB), "r"(block_col), "r"(bar));
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < chunks) issue(s, s);

    for (int chunk = 0; chunk < chunks; chunk++) {
        int s = chunk % STAGES; uint32_t par = (chunk / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
        uint32_t af[4][4][4], bf[4][8][2];   // [sub][mt][..], [sub][n][..]
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * AW + sub * KB64 + acblk * 16; ao ^= ((ao >> 7) & 7) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[sub][mt][0]), "=r"(af[sub][mt][1]), "=r"(af[sub][mt][2]), "=r"(af[sub][mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BWB + sub * KB64 + bkblk * 16; bo ^= ((bo >> 7) & 7) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[sub][n][0]), "=r"(bf[sub][n][1]) : "r"(bd));
            }
        }
#pragma unroll
        for (int sub = 0; sub < WK; sub++)
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[sub][mt][0]), "r"(af[sub][mt][1]), "r"(af[sub][mt][2]), "r"(af[sub][mt][3]), "r"(bf[sub][n][0]), "r"(bf[sub][n][1]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int next = chunk + STAGES;
        if (tid == 0 && next < chunks) {
            asm volatile("{\n\t.reg .pred p;\nWE:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WE;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])), "r"(par));
            issue(s, next);
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
    int M=sz,N=sz,Kf=sz,Kb=Kf/2; uint8_t*dA,*dB; __nv_bfloat16*dC;
    cudaMalloc(&dA,(size_t)M*Kb); cudaMalloc(&dB,(size_t)N*Kb); cudaMalloc(&dC,(size_t)M*N*2);
    cudaMemset(dA,0x22,(size_t)M*Kb); cudaMemset(dB,0x22,(size_t)N*Kb);
    alignas(64) CUtensorMap mapA,mapB; mk(&mapA,dA,Kb,M,AW,BM); mk(&mapB,dB,Kb,N,BWB,BN);
    cudaFuncSetAttribute(matmul_fp4,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    dim3 grid(N/BN,M/(2*BM)),block(256);
    for(int i=0;i<5;i++) matmul_fp4<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Kf);
    if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);int it=20;cudaEventRecord(s);
    for(int i=0;i<it;i++) matmul_fp4<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Kf);
    cudaEventRecord(e);cudaEventSynchronize(e);float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=it;
    double gf=2.0*M*N*Kf/(ms/1e3)/1e9;
    __nv_bfloat16*hC=(__nv_bfloat16*)malloc((size_t)M*N*2);cudaMemcpy(hC,dC,(size_t)M*N*2,cudaMemcpyDeviceToHost);
    int wrong=0; for(size_t i=0;i<(size_t)M*N;i++) if(__bfloat162float(hC[i])!=(float)Kf) wrong++;
    printf("matmul_fp4_wide (dense shared-B WK=4 128B-swz): %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (out0=%.0f exp %d)\n",
           M,N,Kf,ms,gf,wrong==0?"PASS":"FAIL",__bfloat162float(hC[0]),Kf);
    free(hC);cudaFree(dA);cudaFree(dB);cudaFree(dC);
}
int main(){ run(2048); run(4096); run(8192); return 0; }
