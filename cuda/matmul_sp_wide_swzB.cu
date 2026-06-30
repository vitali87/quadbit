#include <cstring>
// Raw-PTX: WIDE-TMA sparse FP4 matmul. The narrow per-step TMA boxes (32B/64B inner) only
// extracted 3.96 TB/s of L2 (54% of the 7.3 TB/s ceiling) -> the "2012k roofline" was an
// artifact. Loading 2 k128-slices per TMA (64B/128B inner) hits 7.13 TB/s, so load time
// (~0.30ms) now matches compute (~0.30ms) and the kernel can approach the mma ceiling
// (~3600k). Each loaded chunk = 2 k128; the loop does 2 mma rounds per chunk. Shared-B
// 256x128, full/empty pipeline, bf16 out. All-ones => out == Klog/2.
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 128
#define BN 128
#define WK 2                 // k128 slices per TMA load
#define AROWB 32
#define BROWB 64
#define AW (AROWB * WK)      // 64  wide A bytes/row
#define BW_ (BROWB * WK)     // 128 wide B bytes/row
#define STAGES 3
#define ASZ (BM * AW)        // 8192 per A-half
#define BSZ (BN * BW_)       // 16384 shared B
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + 2 * STAGES * 8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          __nv_bfloat16 *C, int M, int N, int Klog) {
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
    int lsteps = Klog / (128 * WK);   // wide load-steps

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
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    auto issue = [&](int s, int ls) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])), "l"(&mapA), "r"(ls * AW), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])), "l"(&mapA), "r"(ls * AW), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(ls * BW_), "r"(block_col), "r"(bar));
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < lsteps) issue(s, s);

    for (int ls = 0; ls < lsteps; ls++) {
        int s = ls % STAGES; uint32_t par = (ls / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
        uint32_t af[WK][4][4], bf[WK][8][4];
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * AW + sub * AROWB + acblk * 16;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[sub][mt][0]), "=r"(af[sub][mt][1]), "=r"(af[sub][mt][2]), "=r"(af[sub][mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BW_ + sub * BROWB + bsub * 16; bo ^= ((bo >> 7) & 7) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bf[sub][n][0]), "=r"(bf[sub][n][1]), "=r"(bf[sub][n][2]), "=r"(bf[sub][n][3]) : "r"(bd));
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
                        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[sub][mt][0]), "r"(af[sub][mt][1]), "r"(af[sub][mt][2]), "r"(af[sub][mt][3]),
                          "r"(bf[sub][n][0]), "r"(bf[sub][n][1]), "r"(bf[sub][n][2]), "r"(bf[sub][n][3]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int next = ls + STAGES;
        if (tid == 0 && next < lsteps) {
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
static void mk(CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo, CUtensorMapSwizzle sw) {
    uint64_t gd[2]={(uint64_t)inner,(uint64_t)outer}; uint64_t gs[1]={(uint64_t)inner}; uint32_t bd[2]={(uint32_t)bi,(uint32_t)bo},es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,p,gd,gs,bd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,sw,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("map fail %s\n",s);}
}
static float dfp4v(uint8_t n){int sg=(n>>3)&1,e=(n>>1)&3,m=n&1;float v=(e==0)?(m?0.5f:0.f):((float)(1<<(e-1))*(1.f+0.5f*m));return sg?-v:v;}
void run(int sz, bool ref){
    int M=sz,N=sz,Klog=sz,KAb=Klog/4,KBb=Klog/2; uint8_t*dA,*dB; __nv_bfloat16*dC;
    cudaMalloc(&dA,(size_t)M*KAb); cudaMalloc(&dB,(size_t)N*KBb); cudaMalloc(&dC,(size_t)M*N*2);
    uint8_t*hAc=(uint8_t*)malloc((size_t)M*KAb),*hB=(uint8_t*)malloc((size_t)N*KBb);
    uint8_t*hAlog=ref?(uint8_t*)calloc((size_t)M*Klog,1):nullptr;
    uint32_t st=0x5A5Au; auto rnd=[&]{st^=st<<13;st^=st>>17;st^=st<<5;return st;};
    if(ref){ for(int i=0;i<M;i++)for(int cs=0;cs<KAb;cs++){int pp=(cs/2)*4+(cs%2);uint8_t lo=rnd()&0xf,hi=rnd()&0xf;
        hAc[(size_t)i*KAb+cs]=lo|(hi<<4);hAlog[(size_t)i*Klog+2*pp]=lo;hAlog[(size_t)i*Klog+2*pp+1]=hi;}
        for(size_t i=0;i<(size_t)N*KBb;i++)hB[i]=(uint8_t)rnd(); }
    else { memset(hAc,0x22,(size_t)M*KAb); memset(hB,0x22,(size_t)N*KBb); }
    cudaMemcpy(dA,hAc,(size_t)M*KAb,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB,(size_t)N*KBb,cudaMemcpyHostToDevice);
    alignas(64) CUtensorMap mapA,mapB; mk(&mapA,dA,KAb,M,AW,BM,CU_TENSOR_MAP_SWIZZLE_NONE); mk(&mapB,dB,KBb,N,BW_,BN,CU_TENSOR_MAP_SWIZZLE_128B);
    cudaFuncSetAttribute(matmul_sp,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    dim3 grid(N/BN,M/(2*BM)),block(256);
    for(int i=0;i<5;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Klog);
    if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);int it=20;cudaEventRecord(s);
    for(int i=0;i<it;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dC,M,N,Klog);
    cudaEventRecord(e);cudaEventSynchronize(e);float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=it;
    double gf=2.0*M*N*Klog/(ms/1e3)/1e9;
    __nv_bfloat16*hC=(__nv_bfloat16*)malloc((size_t)M*N*2); cudaMemcpy(hC,dC,(size_t)M*N*2,cudaMemcpyDeviceToHost);
    if(ref){ int wrong=0; float maxrel=0;
        for(int i=0;i<M&&wrong<8;i++)for(int j=0;j<N;j++){ float r=0;
            for(int k=0;k<Klog;k++){ if(((k/2)&3)>=2)continue; uint8_t bn=hB[(size_t)j*KBb+k/2];
                r+=dfp4v(hAlog[(size_t)i*Klog+k])*dfp4v(k&1?bn>>4:bn&0xf); }
            float g=__bfloat162float(hC[(size_t)i*N+j]); float rl=fabsf(g-r)/(fabsf(r)+1.f); if(rl>maxrel)maxrel=rl;
            if(rl>1e-2f){if(wrong<4)printf("  [%d][%d] g %.2f r %.2f\n",i,j,g,r);wrong++;} }
        printf("matmul_sp_wide_swzB VERIFY %dx%dx%d: %s (maxrel %.4f)\n",sz,sz,sz,wrong==0?"PASS":"FAIL",maxrel);
    } else {
        int wrong=0; for(size_t i=0;i<(size_t)M*N;i++) if(__bfloat162float(hC[i])!=(float)(Klog/2)) wrong++;
        printf("matmul_sp_wide_swzB %dx%dx%d: %.3f ms  %.1f GFLOP/s  %s\n",sz,sz,sz,ms,gf,wrong==0?"PASS":"FAIL");
    }
    free(hAc);free(hB);free(hC); if(hAlog)free(hAlog); cudaFree(dA);cudaFree(dB);cudaFree(dC);
}
int main(){ run(512,true); run(1024,true); run(2048,false); run(4096,false); run(8192,false); return 0; }
