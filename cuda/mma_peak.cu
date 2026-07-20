// Hardware ceiling probe: pure tensor-core throughput, NO memory traffic. Fragments
// live in registers; a tight loop issues only mma (32 independent acc tiles => ILP hides
// latency, mirroring the real kernel's 4x8 schedule). Measures whether sm_120 runs
// mma.sp m16n8k128 at the dense-k64 instruction rate (=> real 2x sparse speedup, our
// 1.65M leaves 2x on the table) or at the dense-k128 rate (=> no FLOP speedup, 1.65M is
// near the ceiling). Reports logical GFLOP/s for each: sparse 2*16*8*128/mma.sp, dense
// 2*16*8*64/mma.sync. Both fill the GPU; ITER large so launch/store cost is negligible.

#include <cstdio>
#include <cstdint>

#define ITER 8192
#define NT 32            // independent acc tiles per warp (matches real kernel ILP)

__global__ void __launch_bounds__(256) peak_sp(float *C) {
    uint32_t af[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
    uint32_t bf[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
    uint32_t sa = 0x38383838u, sb = 0x38383838u, meta = 0x44444444u;
    uint16_t z = 0;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int it = 0; it < ITER; it++)
#pragma unroll
        for (int n = 0; n < NT; n++) {
            float d0, d1, d2, d3;
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                  "r"(bf[0]), "r"(bf[1]), "r"(bf[2]), "r"(bf[3]),
                  "f"(acc[n][0]), "f"(acc[n][1]), "f"(acc[n][2]), "f"(acc[n][3]),
                  "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
            acc[n][0] = d0; acc[n][1] = d1; acc[n][2] = d2; acc[n][3] = d3;
        }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NT; i++) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (threadIdx.x == 0) C[blockIdx.x] = s;
}

__global__ void __launch_bounds__(256) peak_dense(float *C) {
    uint32_t af[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
    uint32_t bf[2] = {0x22222222u, 0x22222222u};
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int it = 0; it < ITER; it++)
#pragma unroll
        for (int n = 0; n < NT; n++) {
            float d0, d1, d2, d3;
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16}, %17,{%18,%19};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]),
                  "f"(acc[n][0]), "f"(acc[n][1]), "f"(acc[n][2]), "f"(acc[n][3]),
                  "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
            acc[n][0] = d0; acc[n][1] = d1; acc[n][2] = d2; acc[n][3] = d3;
        }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NT; i++) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (threadIdx.x == 0) C[blockIdx.x] = s;
}

// FED probe: the SAME mma.sp stream as peak_sp, but each iteration pays the REAL feed the kernel pays --
// 4 af + 8 bf ldmatrix from swizzled smem addresses + per-mma metadata/scale register operands. Single warp
// role (every warp loads AND issues mma), acc[32][4] => 128 acc/thread = real 255-reg occupancy. Isolates
// whether the ldmatrix + address-swizzle ALU feed (NOT the tensor core) is what drops us from ~3738 to ~1020.
// smem is filled once (garbage) and reused every iter: we time the compute feed, not memory traffic. If
// peak_fed ~= our 1020 kernel floor => feed is the entire gap => single-role exhausted, warp-spec is the lever.
#define FED_ITER 512
#define AWf 64          // A row bytes in smem (128 fp4 packed)
#define BWf 64
__global__ void __launch_bounds__(256) peak_fed(float *C) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t *a_s = smem;              // 128 rows * AWf = 8KB
    uint8_t *b_s = smem + 128 * AWf;  // 128 rows * BWf = 8KB
    int lane = threadIdx.x & 31, warp = (threadIdx.x >> 5) & 1;  // reuse 2 warp-cols for variety
    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = warp * 32 + j * 8;
    uint32_t sav = 0x38383838u, sbv = 0x38383838u, ev = 0x44444444u;
    uint16_t z = 0;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    // touch smem so ldmatrix reads are valid (not measured)
    for (int i = threadIdx.x; i < (128 * AWf + 128 * BWf) / 4; i += 256) ((uint32_t *)smem)[i] = 0x22222222u;
    __syncthreads();

    for (int it = 0; it < FED_ITER; it++) {
        uint32_t af[4][4], bf[8][4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int ao = ((a_rowt[mt] + arow) & 127) * AWf + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
            uint32_t ad = __cvta_generic_to_shared(&a_s[ao]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
        }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int bo = ((b_col[n] + nrow) & 127) * BWf + bsub * 16; bo ^= ((bo >> 7) & 7) << 4;
            uint32_t bd = __cvta_generic_to_shared(&b_s[bo]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int idx = mt * 8 + n;
                asm volatile(
                    "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13},{%14,%15}, {%16},{%17,%18};"
                    : "+f"(acc[idx][0]), "+f"(acc[idx][1]), "+f"(acc[idx][2]), "+f"(acc[idx][3])
                    : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                      "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                      "r"(ev), "r"(sav), "h"(z), "h"(z), "r"(sbv), "h"(z), "h"(z));
            }
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NT; i++) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (threadIdx.x == 0) C[blockIdx.x] = s;
}

template <class K> static double bench(K k, int blocks, double flop_per_mma) {
    float *dC; cudaMalloc(&dC, blocks * sizeof(float));
    for (int i = 0; i < 3; i++) k<<<blocks, 256>>>(dC);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 10;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) k<<<blocks, 256>>>(dC);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double total_mma = (double)blocks * 8 /*warps*/ * ITER * NT;
    double gflops = total_mma * flop_per_mma / (ms / 1e3) / 1e9;
    cudaFree(dC);
    return gflops;
}

static double bench_fed(int blocks) {
    int smem = 128 * AWf + 128 * BWf;  // 16KB
    cudaFuncSetAttribute(peak_fed, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    float *dC; cudaMalloc(&dC, blocks * sizeof(float));
    for (int i = 0; i < 3; i++) peak_fed<<<blocks, 256, smem>>>(dC);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 10;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) peak_fed<<<blocks, 256, smem>>>(dC);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double total_mma = (double)blocks * 8 * FED_ITER * NT;  // 8 warps, NT=32 mma/iter
    double gflops = total_mma * (2.0 * 16 * 8 * 128) / (ms / 1e3) / 1e9;
    cudaFree(dC);
    return gflops;
}

int main() {
    int blocks = 188 * 4;  // saturate the GPU
    double sp = bench(peak_sp, blocks, 2.0 * 16 * 8 * 128);
    double dn = bench(peak_dense, blocks, 2.0 * 16 * 8 * 64);
    double fed = bench_fed(blocks);
    printf("mma_peak (register-only, no memory):\n");
    printf("  sparse mma.sp  m16n8k128 : %.1f GFLOP/s (logical)\n", sp);
    printf("  dense  mma.sync m16n8k64 : %.1f GFLOP/s\n", dn);
    printf("  sparse/dense ratio       : %.3fx  (2.0 => true 2x sparsity speedup)\n", sp / dn);
    printf("  sparse+FEED (ldmatrix+swizzle, single-role, 128 acc) : %.1f GFLOP/s (%.1f%% of register-only sparse)\n", fed, 100.0 * fed / sp);
    return 0;
}
