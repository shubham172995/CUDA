#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>

#define maskCols 3
#define maskRows 3
#define imgchannels 1

using namespace std;
void sequentialConvolution(const unsigned char*inputImage,const float * kernel ,unsigned char * outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels)
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
}
__global__ void convKernel(unsigned char * InputImageData, const float * kernel,
        unsigned char* outputImageData, int channels, int width, int height){


}

int main(){
    int width, height, bpp;
    unsigned char *img, *seq_img;

    const unsigned char* image = stbi_load( "image64.png", &width, &height, &bpp, imgchannels );
    img = (unsigned char*)malloc(width*height*sizeof(unsigned char));
    seq_img = (unsigned char*)malloc(width*height*sizeof(unsigned char));

    cout << "height " << height << " " << width << std::endl; 



    float hostMaskData[maskRows*maskCols];
    for(int i=0; i< maskCols*maskCols; i++){
        hostMaskData[i] = 1.0/(maskRows*maskCols);
    }
sequentialConvolution(image, hostMaskData, seq_img, maskRows, maskCols, width,
                          height, imgchannels);

stbi_write_png("mynew_seq.png", width, height, imgchannels, seq_img, 0);


    /***************************************/

    // Add cuda code here
    return 0;
}