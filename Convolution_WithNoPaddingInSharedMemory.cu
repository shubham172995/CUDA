__global__ void Convolution(char* image, int* kernel, char* convolutedoutput, int width, int kernelsize){
	__shared__ char subimg[blockDim.y][blockDim.x];
	__shared__ int kernelmatrix[kernelsize][kernelsize];

	int tx, ty, bx, by;
	bx=blockIdx.x;
	by=blockIdx.y;
	tx=threadIdx.x;
	ty=threadIdx.y;

	int row=by*blockDim.y + ty;
	int col=bx*blockDim.x + tx;

	if(row<width && col<width){
		subimg[tx][ty]=image[row*width + col];
	}

	if(tx<kernelsize && ty<kernelsize){
		kernelmatrix[tx][ty]=kernel[(kernelsize-1-ty)*kernelsize + kernelsize-1-tx];	//Flipped in both X and Y direction.
	}

	__syncthreads();

	int anchor=kernelsize/2;

	int temp=0;

	for(int i=ty-anchor;i<=ty+anchor;i++){
		if(i<0){
			if(row+i<0)
				continue;
		}
		if(i>=blockDim.y){
			if(row+i-ty>=width)
				continue;
		}
		for(int j=tx-anchor;j<=tx+anchor;j++){
			if(i<0 && j<0){
				if(col+j<0)
					continue;
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			else if(i<0 && j<blockDim.x){
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			else if(i<0 && j>=blockDim.x){
				if(col+j-tx>=width)
					continue;
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			else if(i<blockDim.y && j<0){
				if(col+j<0)
					continue;
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			else if(i<blockDim.y && j<blockDim.x){
				temp+=subimg[i][j];
			}
			else if(i<blockDim.y && j>=blockDim.x){
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			else if(i>=blockDim.y && j<0){
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			else if(i>=blockDim.y && j<blockDim.x){
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
			if(i>=blockDim.y && j>=blockDim.x){
				if(col+j-tx>=width)
					continue;
				temp+=image[(row+i-ty)*width+col+j-tx]*kernelmatrix[anchor+i][anchor+j];
			}
		}
	}
	convolutedoutput[row*width+col]=temp;
}