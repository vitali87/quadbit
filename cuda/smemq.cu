#include <cstdio>
int main(){
    int v=0,perSM=0,maxblk=0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaDeviceGetAttribute(&perSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&maxblk, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("smem per block (optin) = %d bytes (%.1f KB)\n", v, v/1024.0);
    printf("smem per SM            = %d bytes (%.1f KB)\n", perSM, perSM/1024.0);
    printf("smem per block default = %d bytes (%.1f KB)\n", maxblk, maxblk/1024.0);
    return 0;
}
