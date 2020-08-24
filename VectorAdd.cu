#include<stdio.h>
#include<cuda_runtime.h>

__device__ void warpRed(volatile int*, int);

__global__ void VectorAdd(const int* a, int n){
        extern __shared__ int sa[];

        int tx=threadIdx.x;

        int idx=blockDim.x * blockIdx.x + tx;

        if(idx<n){
                sa[tx]=a[idx];
        }

        __syncthreads();

        for(int stride=blockDim.x>>1;stride>=32;stride>>=1){
                if(tx<stride)
                	sa[tx]+=sa[tx+stride];
                __syncthreads();
        }

        __syncthreads();

        if(tx<32){
                warpRed(sa, tx);
        }
}

__device__ void warpRed(volatile int* sa, int tx){
        sa[tx]+=sa[tx+32];
        sa[tx]+=sa[tx+16];
        sa[tx]+=sa[tx+8];
        sa[tx]+=sa[tx+4];
        sa[tx]+=sa[tx+2];
        sa[tx]+=sa[tx+1];
}

int main(){
        int n=50000;
        cudaError_t err=cudaSuccess;
        size_t size =n*sizeof(int);
        printf("[Size addition of %d elements]\n", n);

        int* h_a=(int*)malloc(size);

        if(h_a==NULL){
            fprintf(stderr, "Failed to allocate vectors\n");
            exit(EXIT_FAILURE);
        }

        int* temp=(int*)malloc(size);

        for(int i=0;i<n;i++){
                h_a[i]=rand()/(int)INT_MAX;
                temp[i]=h_a[i];
        }

        int* d_a=NULL;
        err=cudaMalloc((void**)&d_a, size);

        err=cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

        int threadsPerBlock=256;
        int blocksPerGrid=( n+threadsPerBlock-1 )/threadsPerBlock;

        VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, n);

        err=cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

        for(int i=1;i<n;i++){
                temp[0]+=temp[i];
        }

        if(h_a[0]==temp[0]){
                printf("Test Passed\n");
        }

        err=cudaFree(d_a);

        free(h_a);
        free(temp);

        printf("Done\n");
        return 0;
}
