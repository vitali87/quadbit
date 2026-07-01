#include <cstdio>
#include <cuda.h>
int main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
printf("%s SMs=%d smemPerSM=%zu smemOptin=%d regsPerSM=%d\n",p.name,p.multiProcessorCount,p.sharedMemPerMultiprocessor,p.sharedMemPerBlockOptin,p.regsPerMultiprocessor);
return 0;}
