// Real-scale correctness for the SPARSE block-scaled FP4 mma.sp: random pair-granular
// 2:4 A, random B, AND random per-block ue4m3 scales -- the deployability proof (prior
// verifies were unit-scale only). Scales loaded per the derived scale_vec::4X layout:
//   scale_A[row r][kb] -> lane (r&7)*4+(r>>3), byte kb   (kb = compressed-nonzero K / 16)
//   scale_B[col c][kb] -> lane c*4,            byte kb   (kb = full K / 32)
// One warp, one m16n8k128 tile, f32 out (exact dyadic arithmetic => maxrel ~0 if layout
// is right). metadata 0x44 (pairs {0,1} of each 4 => fp4 mask k%8<4).

#include <cstdio>
#include <cstdint>

#define AROWB 32   // compressed A bytes/row (64 nonzero fp4)
#define BROWB 64   // full B bytes/row (128 fp4)

__global__ void verify(const uint8_t *A, const uint8_t *B, const uint8_t *scA, const uint8_t *scB, float *out) {
    __shared__ __align__(16) uint8_t a_s[16 * AROWB];
    __shared__ __align__(16) uint8_t b_s[8 * BROWB];
    int lane = threadIdx.x & 31;
    for (int i = lane; i < 16 * AROWB; i += 32) a_s[i] = A[i];
    for (int i = lane; i < 8 * BROWB; i += 32) b_s[i] = B[i];
    __syncwarp();

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    uint32_t aa = __cvta_generic_to_shared(&a_s[arow * AROWB + acblk * 16]);
    uint32_t ba = __cvta_generic_to_shared(&b_s[nrow * BROWB + bsub * 16]);
    uint32_t a0, a1, a2, a3, bf0, bf1, bf2, bf3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(aa));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(bf0), "=r"(bf1), "=r"(bf2), "=r"(bf3) : "r"(ba));

    // scale_A: lane L holds row r=(L&3)*8+(L>>2)'s 4 block-scales in bytes 0..3 (r<16)
    int ra = (lane & 3) * 8 + (lane >> 2);
    uint32_t sa = 0x38383838u;
    if (ra < 16) sa = scA[ra * 4 + 0] | (scA[ra * 4 + 1] << 8) | (scA[ra * 4 + 2] << 16) | (scA[ra * 4 + 3] << 24);
    // scale_B: lane L (L%4==0) holds col c=L>>2's 4 block-scales in bytes 0..3
    uint32_t sb = 0x38383838u;
    if ((lane & 3) == 0) { int c = lane >> 2; sb = scB[c * 4 + 0] | (scB[c * 4 + 1] << 8) | (scB[c * 4 + 2] << 16) | (scB[c * 4 + 3] << 24); }
    uint32_t meta = 0x44444444u;
    uint16_t z = 0;

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf0), "r"(bf1), "r"(bf2), "r"(bf3),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));

    int r = lane >> 2, c = (lane & 3) * 2;
    out[r * 8 + c] = d0;
    out[r * 8 + c + 1] = d1;
    out[(r + 8) * 8 + c] = d2;
    out[(r + 8) * 8 + c + 1] = d3;
}

static float dfp4(uint8_t n) {
    int s = (n >> 3) & 1, e = (n >> 1) & 3, m = n & 1;
    float v = (e == 0) ? (m ? 0.5f : 0.f) : ((float)(1 << (e - 1)) * (1.f + 0.5f * m));
    return s ? -v : v;
}
static float due4m3(uint8_t n) {  // ue4m3: e4m3, bias 7
    int s = (n >> 7) & 1, e = (n >> 3) & 0xf, m = n & 7;
    float v = (e == 0) ? ((float)m * 0.001953125f /* 2^-9 */) : (float)(1 + m / 8.0) * exp2f((float)(e - 7));
    return s ? -v : v;
}

int main() {
    int Klog = 128, KAb = Klog / 4, KBb = Klog / 2;  // 32, 64
    uint8_t hA[16 * AROWB], hB[8 * BROWB], hAlog[16 * 128], hScA[16 * 4], hScB[8 * 4];
    uint32_t st = 0x1234u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (int i = 0; i < 16 * 128; i++) hAlog[i] = 0;
    // A: pair-granular 2:4 compressed (cs = selected pair index), record logical nonzero
    for (int i = 0; i < 16; i++)
        for (int cs = 0; cs < KAb; cs++) {
            int pp = (cs / 2) * 4 + (cs % 2);  // logical pair (selected pp%4 in {0,1})
            uint8_t lo = rnd() & 0xf, hi = rnd() & 0xf;
            hA[i * AROWB + cs] = lo | (hi << 4);
            hAlog[i * 128 + 2 * pp + 0] = lo;
            hAlog[i * 128 + 2 * pp + 1] = hi;
        }
    for (int i = 0; i < 8 * BROWB; i++) hB[i] = (uint8_t)rnd();
    // scales: ue4m3 bytes, sign 0, exp in [6,8] (~0.5..~3.75), random mantissa
    for (int i = 0; i < 16 * 4; i++) hScA[i] = ((6 + (rnd() % 3)) << 3) | (rnd() & 7);
    for (int i = 0; i < 8 * 4; i++) hScB[i] = ((6 + (rnd() % 3)) << 3) | (rnd() & 7);

    uint8_t *dA, *dB, *dScA, *dScB;
    float *dO, hO[16 * 8];
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dScA, sizeof(hScA)); cudaMalloc(&dScB, sizeof(hScB)); cudaMalloc(&dO, sizeof(hO));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemcpy(dScA, hScA, sizeof(hScA), cudaMemcpyHostToDevice);
    cudaMemcpy(dScB, hScB, sizeof(hScB), cudaMemcpyHostToDevice);
    verify<<<1, 32>>>(dA, dB, dScA, dScB, dO);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(hO, dO, sizeof(hO), cudaMemcpyDeviceToHost);

    // reference: sum over nonzero n=0..63 of (Aval*scA_block) * (Bval*scB_block)
    int wrong = 0; float maxrel = 0.f;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++) {
            float ref = 0.f;
            for (int n = 0; n < 64; n++) {
                int cs = n / 2;
                int pp = (cs / 2) * 4 + (cs % 2);
                int k = 2 * pp + (n & 1);                 // logical K of this nonzero
                float av = dfp4(hAlog[i * 128 + k]) * due4m3(hScA[i * 4 + n / 16]);
                uint8_t bn = hB[j * BROWB + k / 2];
                float bv = dfp4((k & 1) ? bn >> 4 : bn & 0xf) * due4m3(hScB[j * 4 + k / 32]);
                ref += av * bv;
            }
            float got = hO[i * 8 + j];
            float rel = fabsf(got - ref) / (fabsf(ref) + 1.f);
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-3f) { if (wrong < 6) printf("  [%d][%d] got %.4f ref %.4f\n", i, j, got, ref); wrong++; }
        }
    printf("sp_verify_scaled (random A/B + random ue4m3 4X scales, 16x8x128): %s (%d wrong, maxrel %.5f)\n",
           wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    return 0;
}
