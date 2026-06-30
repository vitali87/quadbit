// SMALL-SHAPE config: single warp-group (128x128 tile, NO shared-B) so small problems
// produce 4x the CTAs of the 256x128 shared-B kernel and fill the machine (kills the wave
// quantization that drops 2048^3 to 624k). One A TMA + one B TMA per chunk; the freed smem
// (no 2*A) lets us run STAGES=3. Same wide-swizzle load path + full meta + ue4m3 scales.
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 128
#define BN 128
#define WK 2
#define AROWB 32
#define BROWB 64
#define AW (AROWB * WK)
#define BW_ (BROWB * WK)
#define STAGES 3
#define ASZ (BM * AW)
#define BSZ (BN * BW_)
#define SCA 512
#define SCB 512
#define MET 1024
#define SMEM (STAGES*ASZ + STAGES*BSZ + STAGES*WK*SCA + STAGES*WK*SCB + STAGES*WK*MET + 2*STAGES*8 + 128)

__global__ void __launch_bounds__(128)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          const uint8_t *scaleA, const uint8_t *scaleB, const uint32_t *meta,
          __nv_bfloat16 *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem;
    uint8_t *b_s = smem + STAGES * ASZ;
    uint8_t *scA_sm = b_s + STAGES * BSZ;
    uint8_t *scB_sm = scA_sm + STAGES * WK * SCA;
    uint8_t *met_sm = scB_sm + STAGES * WK * SCB;
    uint64_t *full = (uint64_t *)(met_sm + STAGES * WK * MET);
    uint64_t *empty = full + STAGES;

    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    int chunks = Klog / (128 * WK);

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;
    int ra_local = (lane & 3) * 8 + (lane >> 2), cb_local = lane >> 2;
    bool a_valid = ra_local < 16, b_valid = (lane & 3) == 0;
    int a_sidx[4], b_sidx[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_sidx[mt] = a_rowt[mt] + ra_local;
#pragma unroll
    for (int n = 0; n < 8; n++) b_sidx[n] = b_col[n] + cb_local;
    int mma_row = (lane & 1) * 8 + (lane >> 2), Hh = (lane >> 1) & 1;
    int m_sidx[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) m_sidx[mt] = (a_rowt[mt] + mma_row) * 2 + Hh;

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint16_t z = 0;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    auto issue = [&](int s, int chunk) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                     "r"((uint32_t)(ASZ + BSZ + WK * SCA + WK * SCB + WK * MET)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&a_s[s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(chunk * BW_), "r"(block_col), "r"(bar));
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
            int step = chunk * WK + sub;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&scA_sm[(s * WK + sub) * SCA])), "l"(scaleA + (size_t)(step * M + block_row) * 4), "r"((uint32_t)SCA), "r"(bar));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&scB_sm[(s * WK + sub) * SCB])), "l"(scaleB + (size_t)(step * N + block_col) * 4), "r"((uint32_t)SCB), "r"(bar));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&met_sm[(s * WK + sub) * MET])), "l"((const uint8_t *)meta + (size_t)(step * M + block_row) * 8), "r"((uint32_t)MET), "r"(bar));
        }
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < chunks) issue(s, s);

    for (int chunk = 0; chunk < chunks; chunk++) {
        int s = chunk % STAGES; uint32_t par = (chunk / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
            const uint32_t *scA = (const uint32_t *)(scA_sm + (s * WK + sub) * SCA);
            const uint32_t *scB = (const uint32_t *)(scB_sm + (s * WK + sub) * SCB);
            const uint32_t *mtA = (const uint32_t *)(met_sm + (s * WK + sub) * MET);
            uint32_t sav[4], sbv[8], ev[4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) { sav[mt] = a_valid ? scA[a_sidx[mt]] : 0x38383838u; ev[mt] = mtA[m_sidx[mt]]; }
#pragma unroll
            for (int n = 0; n < 8; n++) sbv[n] = b_valid ? scB[b_sidx[n]] : 0x38383838u;
            uint32_t af[4][4], bf[8][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * AW + sub * AROWB + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BW_ + sub * BROWB + bsub * 16; bo ^= ((bo >> 7) & 7) << 4;
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
                          "r"(ev[mt]), "r"(sav[mt]), "h"(z), "h"(z), "r"(sbv[n]), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
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
static void mk(CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo, CUtensorMapSwizzle sw) {
    uint64_t gd[2]={(uint64_t)inner,(uint64_t)outer}; uint64_t gs[1]={(uint64_t)inner}; uint32_t bd[2]={(uint32_t)bi,(uint32_t)bo},es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,p,gd,gs,bd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,sw,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("map fail %s\n",s);}
}
static float dfp4(uint8_t n){int s=(n>>3)&1,e=(n>>1)&3,m=n&1;float v=(e==0)?(m?0.5f:0.f):((float)(1<<(e-1))*(1.f+0.5f*m));return s?-v:v;}
static float due4m3(uint8_t n){int s=(n>>7)&1,e=(n>>3)&0xf,m=n&7;float v=(e==0)?((float)m*0.001953125f):(float)(1+m/8.0)*exp2f((float)(e-7));return s?-v:v;}
void run(int sz, bool ref) {
    int M=sz,N=sz,Klog=sz,KAb=Klog/4,KBb=Klog/2,ksteps=Klog/128;
    uint8_t*dA,*dB,*dScA,*dScB; uint32_t*dMeta; __nv_bfloat16*dC;
    cudaMalloc(&dA,(size_t)M*KAb);cudaMalloc(&dB,(size_t)N*KBb);
    cudaMalloc(&dScA,(size_t)ksteps*M*4);cudaMalloc(&dScB,(size_t)ksteps*N*4);
    cudaMalloc(&dMeta,(size_t)ksteps*M*2*4);cudaMalloc(&dC,(size_t)M*N*2);
    uint8_t*hAc=(uint8_t*)calloc((size_t)M*KAb,1),*hB=(uint8_t*)malloc((size_t)N*KBb);
    uint8_t*hScA=(uint8_t*)malloc((size_t)ksteps*M*4),*hScB=(uint8_t*)malloc((size_t)ksteps*N*4);
    uint32_t*hMeta=(uint32_t*)calloc((size_t)ksteps*M*2,4);
    uint8_t*idx=ref?(uint8_t*)malloc((size_t)M*ksteps*16*2):0;
    uint32_t st=0x71u; auto rnd=[&]{st^=st<<13;st^=st>>17;st^=st<<5;return st;};
    for(int i=0;i<M;i++)for(int step=0;step<ksteps;step++)for(int g=0;g<16;g++){int a=rnd()%4,b;do{b=rnd()%4;}while(b==a);if(a>b){int t=a;a=b;b=t;}
        hMeta[((size_t)step*M+i)*2+g/8]|=(uint32_t)(a|(b<<2))<<((g%8)*4); if(ref){idx[(((size_t)i*ksteps+step)*16+g)*2]=a;idx[(((size_t)i*ksteps+step)*16+g)*2+1]=b;}}
    for(int i=0;i<M;i++)for(int step=0;step<ksteps;step++)for(int cs=0;cs<32;cs++){uint8_t lo=rnd()&0xf,hi=rnd()&0xf;hAc[(size_t)i*KAb+step*32+cs]=lo|(hi<<4);}
    for(size_t i=0;i<(size_t)N*KBb;i++)hB[i]=(uint8_t)rnd();
    for(size_t i=0;i<(size_t)ksteps*M*4;i++)hScA[i]=((6+(rnd()%3))<<3)|(rnd()&7);
    for(size_t i=0;i<(size_t)ksteps*N*4;i++)hScB[i]=((6+(rnd()%3))<<3)|(rnd()&7);
    cudaMemcpy(dA,hAc,(size_t)M*KAb,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,(size_t)N*KBb,cudaMemcpyHostToDevice);
    cudaMemcpy(dScA,hScA,(size_t)ksteps*M*4,cudaMemcpyHostToDevice);cudaMemcpy(dScB,hScB,(size_t)ksteps*N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dMeta,hMeta,(size_t)ksteps*M*2*4,cudaMemcpyHostToDevice);
    alignas(64) CUtensorMap mapA,mapB; mk(&mapA,dA,KAb,M,AW,BM,CU_TENSOR_MAP_SWIZZLE_64B); mk(&mapB,dB,KBb,N,BW_,BN,CU_TENSOR_MAP_SWIZZLE_128B);
    cudaFuncSetAttribute(matmul_sp,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    dim3 grid(N/BN,M/BM),block(128);
    for(int i=0;i<5;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dScA,dScB,dMeta,dC,M,N,Klog);
    if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);int it=20;cudaEventRecord(s);
    for(int i=0;i<it;i++) matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dScA,dScB,dMeta,dC,M,N,Klog);
    cudaEventRecord(e);cudaEventSynchronize(e);float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=it;
    double gf=2.0*M*N*Klog/(ms/1e3)/1e9;
    __nv_bfloat16*hC=(__nv_bfloat16*)malloc((size_t)M*N*2);cudaMemcpy(hC,dC,(size_t)M*N*2,cudaMemcpyDeviceToHost);
    if(ref){int wrong=0;float maxrel=0;
        for(int i=0;i<M&&wrong<8;i++)for(int j=0;j<N;j++){float r=0;
            for(int step=0;step<ksteps;step++)for(int cs=0;cs<32;cs++){int g=cs/2,slot=cs%2;int pidx=idx[(((size_t)i*ksteps+step)*16+g)*2+slot];
                int pair=step*64+g*4+pidx;int gK=2*pair;float sA=due4m3(hScA[((size_t)step*M+i)*4+cs/8]);uint8_t ab=hAc[(size_t)i*KAb+step*32+cs];
                uint8_t bl=hB[(size_t)j*KBb+gK/2],bh=hB[(size_t)j*KBb+(gK+1)/2];
                float Bl=dfp4((gK&1)?bl>>4:bl&0xf)*due4m3(hScB[((size_t)step*N+j)*4+(gK-step*128)/32]);
                float Bh=dfp4(((gK+1)&1)?bh>>4:bh&0xf)*due4m3(hScB[((size_t)step*N+j)*4+((gK+1)-step*128)/32]);
                r+=dfp4(ab&0xf)*sA*Bl+dfp4(ab>>4)*sA*Bh;}
            float g=__bfloat162float(hC[(size_t)i*N+j]);float rl=fabsf(g-r)/(fabsf(r)+1.f);if(rl>maxrel)maxrel=rl;
            if(rl>5e-3f){if(wrong<4)printf("  [%d][%d] g %.2f r %.2f\n",i,j,g,r);wrong++;}}
        printf("matmul_sp_small VERIFY %dx%dx%d: %s (maxrel %.4f)\n",sz,sz,sz,wrong==0?"PASS":"FAIL",maxrel);
    } else printf("matmul_sp_small %dx%dx%d: %.3f ms  %.1f GFLOP/s\n",sz,sz,sz,ms,gf);
    free(hAc);free(hB);free(hScA);free(hScB);free(hMeta);free(hC);if(idx)free(idx);
    cudaFree(dA);cudaFree(dB);cudaFree(dScA);cudaFree(dScB);cudaFree(dMeta);cudaFree(dC);
}
int main(){ run(512,true); run(1024,false); run(2048,false); run(4096,false); run(8192,false); return 0; }
