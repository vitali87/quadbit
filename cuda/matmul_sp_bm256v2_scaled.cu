// Raw-PTX track: 2:4-sparse FP4 matmul, traffic-optimal tiling + decoupled async pipeline.
// The mem-only probe proved the 256x128 shared-B tiling cuts the memory ceiling from
// 1620k -> 2037k (20% less B traffic). But a CTA-wide __syncthreads exposed compute
// (1448k < 2037k ceiling). This version replaces the CTA barrier with a two-mbarrier
// full/empty pipeline: consumers wait `full` (data ready), compute, signal `empty`; the
// producer waits `empty` (buffer free) before refilling. No CTA-wide lockstep, so a fast
// warp runs ahead into the next stage while the producer refills => compute hides behind
// the lower traffic. All-ones => out == Klog/2.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 128
#define BN 128
#define BKL 128
#define AROWB 32
#define BROWB 64
#define STAGES 6
#define ASZ (BM * AROWB)
#define BSZ (BN * BROWB)
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + 2 * STAGES * 8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          const uint8_t *scaleA, const uint8_t *scaleB, float *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;

    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);
    uint64_t *empty = full + STAGES;

    int block_row = blockIdx.y * (2 * BM) + wg * BM;
    int a_load_row = blockIdx.y * (2 * BM);
    int block_col = blockIdx.x * BN;
    int ksteps = Klog / BKL;

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;

    // derived ue4m3 4X layout: lane carries scaleA row (ra<16) / scaleB col (lane%4==0).
    // Precompute each lane's per-tile u32 base index (row/col * ksteps); -1 = inactive.
    const uint32_t *sA32 = (const uint32_t *)scaleA, *sB32 = (const uint32_t *)scaleB;
    int ra_local = (lane & 3) * 8 + (lane >> 2), cb_local = lane >> 2;
    int abase[4], bbase[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) abase[mt] = (ra_local < 16) ? (block_row + a_rowt[mt] + ra_local) * ksteps : -1;
#pragma unroll
    for (int n = 0; n < 8; n++) bbase[n] = ((lane & 3) == 0) ? (block_col + b_col[n] + cb_local) * ksteps : -1;

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t meta = 0x44444444u;
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

    auto issue = [&](int s, int step) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])),
                     "l"(&mapB), "r"(step * BROWB), "r"(block_col), "r"(bar));
    };

    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < ksteps) issue(s, s);

    // double-buffered scale registers: prefetch step+1 while computing step (hides L2 latency)
    uint32_t sav[4], sbv[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) sav[mt] = (abase[mt] >= 0) ? __ldg(sA32 + abase[mt]) : 0x38383838u;
#pragma unroll
    for (int n = 0; n < 8; n++) sbv[n] = (bbase[n] >= 0) ? __ldg(sB32 + bbase[n]) : 0x38383838u;

    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\n\tWAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra WAIT;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));

        // prefetch next step's scales early; independent of ldmatrix/mma so latency overlaps
        uint32_t savn[4], sbvn[8];
        int nstep = step + 1;
        if (nstep < ksteps) {
#pragma unroll
            for (int mt = 0; mt < 4; mt++) savn[mt] = (abase[mt] >= 0) ? __ldg(sA32 + abase[mt] + nstep) : 0x38383838u;
#pragma unroll
            for (int n = 0; n < 8; n++) sbvn[n] = (bbase[n] >= 0) ? __ldg(sB32 + bbase[n] + nstep) : 0x38383838u;
        }

        int aoff = s * ASZ, boff = s * BSZ;
        uint32_t af[4][4], bf[8][4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + (a_rowt[mt] + arow) * AROWB + acblk * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
        }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t bd = __cvta_generic_to_shared(&b_s[boff + (b_col[n] + nrow) * BROWB + bsub * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int idx = mt * 8 + n;
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                      "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                      "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                      "r"(meta), "r"(sav[mt]), "h"(z), "h"(z), "r"(sbv[n]), "h"(z), "h"(z));
                acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
            }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) {
            asm volatile("{\n\t.reg .pred p;\n\tWE:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra WE;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])), "r"(par));
            issue(s, next);
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++) sav[mt] = savn[mt];
#pragma unroll
        for (int n = 0; n < 8; n++) sbv[n] = sbvn[n];
    }

#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2);
            int gc = block_col + b_col[n] + (lane & 3) * 2;
            C[gr * N + gc] = acc[idx][0];
            C[gr * N + gc + 1] = acc[idx][1];
            C[(gr + 8) * N + gc] = acc[idx][2];
            C[(gr + 8) * N + gc + 1] = acc[idx][3];
        }
}

static void mk(const char *tag, CUtensorMap *m, uint8_t *p, int inner, int outer, int boxin, int boxout) {
    uint64_t gdim[2] = {(uint64_t)inner, (uint64_t)outer};
    uint64_t gstride[1] = {(uint64_t)inner};
    uint32_t bdim[2] = {(uint32_t)boxin, (uint32_t)boxout};
    uint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gdim, gstride, bdim, estride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map[%s] failed: %s\n", tag, s); }
}


static float dfp4(uint8_t n){int s=(n>>3)&1,e=(n>>1)&3,m=n&1;float v=(e==0)?(m?0.5f:0.f):((float)(1<<(e-1))*(1.f+0.5f*m));return s?-v:v;}
static float due4m3(uint8_t n){int s=(n>>7)&1,e=(n>>3)&0xf,m=n&7;float v=(e==0)?((float)m*0.001953125f):(float)(1+m/8.0)*exp2f((float)(e-7));return s?-v:v;}

// scaleA[M][ksteps][4], scaleB[N][ksteps][4] ue4m3 (per-row/col, per-k128-step, 4 sub-blocks)
void run(int sz, bool do_ref) {
    int M = sz, N = sz, Klog = sz, KAb = Klog / 4, KBb = Klog / 2, ksteps = Klog / BKL;
    uint8_t *dA, *dB, *dScA, *dScB; float *dC;
    cudaMalloc(&dA, (size_t)M * KAb); cudaMalloc(&dB, (size_t)N * KBb);
    cudaMalloc(&dScA, (size_t)M * ksteps * 4); cudaMalloc(&dScB, (size_t)N * ksteps * 4);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));

    uint8_t *hA=(uint8_t*)malloc((size_t)M*KAb), *hB=(uint8_t*)malloc((size_t)N*KBb);
    uint8_t *hAlog=(uint8_t*)calloc((size_t)M*Klog,1);
    uint8_t *hScA=(uint8_t*)malloc((size_t)M*ksteps*4), *hScB=(uint8_t*)malloc((size_t)N*ksteps*4);
    uint32_t st=0x77u; auto rnd=[&]{st^=st<<13;st^=st>>17;st^=st<<5;return st;};
    for(int i=0;i<M;i++)for(int cs=0;cs<KAb;cs++){int pp=(cs/2)*4+(cs%2);uint8_t lo=rnd()&0xf,hi=rnd()&0xf;
        hA[(size_t)i*KAb+cs]=lo|(hi<<4);hAlog[(size_t)i*Klog+2*pp]=lo;hAlog[(size_t)i*Klog+2*pp+1]=hi;}
    for(size_t i=0;i<(size_t)N*KBb;i++)hB[i]=(uint8_t)rnd();
    for(size_t i=0;i<(size_t)M*ksteps*4;i++)hScA[i]=((6+(rnd()%3))<<3)|(rnd()&7);
    for(size_t i=0;i<(size_t)N*ksteps*4;i++)hScB[i]=((6+(rnd()%3))<<3)|(rnd()&7);
    cudaMemcpy(dA,hA,(size_t)M*KAb,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,(size_t)N*KBb,cudaMemcpyHostToDevice);
    cudaMemcpy(dScA,hScA,(size_t)M*ksteps*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dScB,hScB,(size_t)N*ksteps*4,cudaMemcpyHostToDevice);

    alignas(64) CUtensorMap mapA, mapB;
    mk("A",&mapA,dA,KAb,M,AROWB,BM); mk("B",&mapB,dB,KBb,N,BROWB,BN);
    cudaFuncSetAttribute(matmul_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N/BN, M/(2*BM)), block(256);
    for(int i=0;i<5;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dScA,dScB,dC,M,N,Klog);
    if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}

    cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e); int it=20;
    cudaEventRecord(s); for(int i=0;i<it;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dScA,dScB,dC,M,N,Klog);
    cudaEventRecord(e);cudaEventSynchronize(e); float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=it;
    double gf=2.0*M*N*Klog/(ms/1e3)/1e9;

    if(!do_ref){printf("matmul_sp_bm256v2_scaled (real ue4m3 scales): %dx%dx%d  %.3f ms  %.1f GFLOP/s (timing)\n",M,N,Klog,ms,gf);
        free(hA);free(hB);free(hAlog);free(hScA);free(hScB);cudaFree(dA);cudaFree(dB);cudaFree(dScA);cudaFree(dScB);cudaFree(dC);return;}

    float *hC=(float*)malloc((size_t)M*N*sizeof(float));
    cudaMemcpy(hC,dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost);
    int wrong=0; float maxrel=0.f;
    for(int i=0;i<M&&wrong<8;i++)for(int j=0;j<N;j++){
        float ref=0.f;
        for(int step=0;step<ksteps;step++)for(int n=0;n<64;n++){
            int cs=n/2,pp=(cs/2)*4+(cs%2); int kl=2*pp+(n&1); int gK=step*128+kl;
            float av=dfp4(hAlog[(size_t)i*Klog+gK])*due4m3(hScA[((size_t)i*ksteps+step)*4 + n/16]);
            uint8_t bn=hB[(size_t)j*KBb+gK/2]; float bv=dfp4((gK&1)?bn>>4:bn&0xf)*due4m3(hScB[((size_t)j*ksteps+step)*4 + kl/32]);
            ref+=av*bv;
        }
        float got=hC[(size_t)i*N+j]; float rel=fabsf(got-ref)/(fabsf(ref)+1.f);
        if(rel>maxrel)maxrel=rel;
        if(rel>1e-3f){if(wrong<4)printf("  [%d][%d] got %.3f ref %.3f\n",i,j,got,ref);wrong++;}
    }
    printf("matmul_sp_bm256v2_scaled (real ue4m3 scales): %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (maxrel %.5f)\n",
           M,N,Klog,ms,gf,wrong==0?"PASS":"FAIL",maxrel);
    free(hA);free(hB);free(hAlog);free(hScA);free(hScB);free(hC);cudaFree(dA);cudaFree(dB);cudaFree(dScA);cudaFree(dScB);cudaFree(dC);
}

int main(){ run(512,true); run(2048,false); run(4096,false); run(8192,false); return 0; }
