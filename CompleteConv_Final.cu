#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>

#define maskCols 3
#define maskRows 3
#define imgchannels 1

__constant__ float kernelmatrix[3][3];

using namespace std;

/*void sequentialConvolution(const unsigned char*inputImage,const float * kernel ,unsigned char * outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;                         // center index of kernel
    float sum;                                      // accumulation variable
    int rowIndex, colIndex;                         // indice di riga e di colonna

    const unsigned char * inputImageData = inputImage;
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < dataSizeY; ++i)                //cycle on image rows
        {
            for (j = 0; j < dataSizeX; ++j)            //cycle on image columns
            {
                sum = 0;
                for (m = 0; m < kernelSizeY; ++m)      //cycle kernel rows
                {
                    mm = kernelSizeY - 1 - m;       // row index of flipped kernel

                    for (n = 0; n < kernelSizeX; ++n)  //cycle on kernel columns
                    {
                        nn = kernelSizeX - 1 - n;   // column index of flipped kernel

                        // indexes used for checking boundary
                        rowIndex = i + m - kCenterY;
                        colIndex = j + n - kCenterX;

                        // ignore pixels which are out of bound
                        if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                            sum += inputImageData[(dataSizeX * rowIndex + colIndex)*channels + k] * kernel[kernelSizeX * mm + nn];
                    }
                }
                outputImageData[(dataSizeX * i + j)*channels + k] = sum;

            }
        }
    }
}*/


__global__ void Convolution(unsigned char* image, unsigned char* convolutedoutput, int width, int kernelsize){

    int anchor=kernelsize/2;

    __shared__ char subimg[18][18];

    int tx, ty, bx, by;
    bx=blockIdx.x;
    by=blockIdx.y;
    tx=threadIdx.x;
    ty=threadIdx.y;

    int row=by*blockDim.y + ty;
    int col=bx*blockDim.x + tx;

    if(row<width && col<width){
        subimg[ty + anchor][tx + anchor]=image[row*width + col];
    }
    __syncthreads();


    if(row<anchor){
        subimg[ty][tx]=0;
        subimg[ty][tx+anchor]=0;
    }

    if(col<anchor){
        subimg[ty][tx]=0;
        subimg[ty+anchor][tx]=0;
    }

    if(width-row-1<anchor){
        subimg[ty+2*anchor][tx]=0;
        subimg[ty+2*anchor][tx+anchor]=0;
    }


    if(width-col-1<anchor){
        subimg[ty][tx+2*anchor]=0;
        subimg[ty+anchor][tx+2*anchor]=0;
    }

    if(ty<anchor && by>=1){
        subimg[ty][tx+anchor]=image[(row-anchor)*width+col];
        if(tx<anchor && bx>=1){
            subimg[ty][tx]=image[(row-anchor)*width+col-anchor];
        }
        if((tx+anchor)>(blockDim.x-1) && bx<(width/blockDim.x-1)){
            subimg[ty][tx+(2*anchor)]=image[(row-anchor)*width+col+anchor];
        }
    }

    if(tx<anchor && bx>=1){
        subimg[ty+anchor][tx]=image[row*width+col-anchor];
    }

    if(bx<(width/blockDim.x-1) && (tx+anchor>=(blockDim.x))){
        subimg[ty+anchor][tx+(2*anchor)]=image[row*width+col+anchor];
    }

    if(by<(width/blockDim.y-1) && ((ty+anchor)>=blockDim.y)){
        subimg[ty+(2*anchor)][tx+anchor]=image[(row+anchor)*width+col];
        if(tx<anchor && bx>=1){
            subimg[ty+(2*anchor)][tx]=image[(row+anchor)*width+col-anchor];
        }
        if((tx+anchor)>(blockDim.x-1) && bx<(width/blockDim.x-1)){
            subimg[ty+(2*anchor)][tx+(2*anchor)]=image[(row+anchor)*width+col+anchor];
        }
    }

    __syncthreads();

    int temp=0;

    int r=ty+anchor, c=tx+anchor;
    int kernelrow=0, kernelcol=0;

    for(int i=r-anchor;i<=r+anchor;i++){
        kernelcol=0;
        for(int j=c-anchor;j<=c+anchor;j++){
            temp+=subimg[i][j]*kernelmatrix[kernelrow][kernelcol];
            ++kernelcol;
        }
        ++kernelrow;
    }

    __syncthreads();

    convolutedoutput[row*width+col]=temp;
}

int main(){
    int width, height, bpp, err;
    unsigned char *seq_img, *d_img, *d_output;

    const unsigned char* image = stbi_load( "image64.png", &width, &height, &bpp, imgchannels );
    //img = (unsigned char*)malloc(width*height*sizeof(unsigned char));
    seq_img = (unsigned char*)malloc(width*height*sizeof(unsigned char));

    err=cudaMalloc((void**)&d_img, width*height*sizeof(unsigned char));
    err=cudaMemcpy(d_img, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    err=cudaMalloc((void**)&d_output, width*height*sizeof(unsigned char));

    cout << "height " << height << " " << width << std::endl; 

    dim3 dimBlock(4,4);
    dim3 dimGrid(16,16);

    float kernel[maskRows][maskCols], kernel1[maskRows][maskCols];
    for(int i=0; i< maskCols; i++){
        for(int j=0;j<maskCols;j++){
            kernel[i][j] = 1.0/(maskRows*maskCols);
            kernel1[i][j] = kernel[i][j];
        }
    }

    for(int i=0; i< maskCols; i++){
        for(int j=0;j<maskCols;j++){
            kernel[i][j] = kernel1[maskRows-i-1][maskCols-j-1];
        }
    }

    err=cudaMemcpyToSymbol(kernelmatrix, kernel, maskRows*maskCols*sizeof(float));

    Convolution<<<dimBlock, dimGrid>>>(d_img, d_output, width, maskRows);

    err=cudaMemcpy(seq_img, d_output, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

//sequentialConvolution(image, hostMaskData, seq_img, maskRows, maskCols, width, height, imgchannels);

    stbi_write_png("mynew_seq.png", width, height, imgchannels, seq_img, 0);
    stbi_write_png("mynew_seq1.png", width, height, imgchannels, image, 0);


    /***************************************/

    // Add cuda code here
    return 0;
}