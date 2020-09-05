/*****************************************************************************
*
* String Pattern Matching - Serial Implementation
* 
* Reference: http://people.maths.ox.ac.uk/~gilesm/cuda/
*
*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#define LINEWIDTH 20


__constant__ unsigned int wordlist[32];


void matchPattern_CPU(unsigned int *text, unsigned int *words, int *matches, int nwords, int length)
{
	unsigned int word;

	printf("CPU    %d %d\n", (text[1]>>(8*1)) + (text[1+1]<<(32-8*1)), (text[0]>>(8*1)) + (text[1]<<(32-8*1)));

	for (int l=0; l<length; l++)
	{
		for (int offset=0; offset<4; offset++)
		{
			if (offset==0)
				word = text[l];
			else
				word = (text[l]>>(8*offset)) + (text[l+1]<<(32-8*offset)); 

			for (int w=0; w<nwords; w++){
				matches[w] += (word==words[w]);
			} 	
		}
	}
}



__global__ void matchPattern_GPU(unsigned int* text, unsigned int* matches, int nwords, int length){

	__shared__ unsigned int s_words[513];


	unsigned int matches1[32];
	memset(matches1, 0, sizeof(matches1));
	int bx=blockIdx.x;
	int tx=threadIdx.x;
	unsigned int word=0;
	int idx=bx*blockDim.x+tx;

	if(idx<length)
		s_words[tx]=text[idx];

	__syncthreads();

	/*if(tx==0){
		if(idx==0)
			s_words[tx]=0;
		else
			s_words[tx]=text[idx-1];
	}*/

	if(tx==blockDim.x-1){
		if(idx<length)
			s_words[tx+1]=text[idx+1];
	}

	__syncthreads();

	if(idx==1){
		printf("HOLL   %d %d\n",(s_words[tx+1]>>(8*1)) + (s_words[tx+2]<<(32-8*1)), (s_words[tx]>>(8*1)) + (s_words[tx+1]<<(32-8*1)));
	}

	for (int offset=0; offset<4; offset++){
		if (offset==0)
			word = s_words[tx];
		else
			word = (s_words[tx]>>(8*offset)) + (s_words[tx+1]<<(32-8*offset));
		for (int w=0; w<nwords; w++){
			if(word==wordlist[w])
				atomicAdd(&matches[w], 1);
		} 	
	}

	/*__syncthreads();

	for(int i=0;i<32;i++){
		atomicAdd(&matches[i], matches1[i]);
	}*/

	/*int idx=blockDim.x*blockIdx.x+threadIdx.x;
	if(idx==1485424)
		printf("Block number   %d\n", blockIdx.x);
	unsigned int word=0;
	unsigned int matches1[32];
	memset(matches1, 0, sizeof(matches1));
	if(idx<length){
		for (int offset=0; offset<4; offset++){
			if (offset==0)
				word = text[idx];
			else
				word = (text[idx]>>(8*offset)) + (text[idx+1]<<(32-8*offset)); 
			for (int w=0; w<nwords; w++){
				matches1[w] += (word==wordlist[w]);
			} 	
		}
	}

	for(int i=0;i<32;i++){
		matches[i]+=matches1[i];
	}*/	
}



int main(int argc, const char **argv)
{

	int length, err, len, nwords=32, matches[nwords], matches1[nwords];
	char *ctext, keywords[nwords][LINEWIDTH], *line;
	line = (char*) malloc(sizeof(char)*LINEWIDTH);
	unsigned int  *text,  *words;
	memset(matches, 0, sizeof(matches));
	memset(matches1, 0, sizeof(matches1));

	unsigned int* d_text, *d_matches;


	// read in text and keywords for processing
	FILE *fp, *wfile;
	wfile = fopen("./keywords.txt","r");
	if (!wfile)
	{	printf("keywords.txt: File not found.\n");	exit(0);}

	int k=0, cnt = nwords;
	size_t read, linelen = LINEWIDTH;
	while((read = getline(&line, &linelen, wfile)) != -1 && cnt--)
	{
		strncpy(keywords[k], line, sizeof(line));
		keywords[k][4] = '\0';
		k++;
	}
	fclose(wfile);



	fp = fopen("./medium.txt","r");
	if (!fp)
	{	printf("Unable to open the file.\n");	exit(0);}

	length = 0;
	while (getc(fp) != EOF) length++;
	ctext = (char *) malloc(length+4);

	rewind(fp);

	for (int l=0; l<length; l++) ctext[l] = getc(fp);
	for (int l=length; l<length+4; l++) ctext[l] = ' ';

	fclose(fp);

	printf("Length : %d\n", length );
	// define number of words of text, and set pointers
	len  = length/4;
	text = (unsigned int *) ctext;

	// define words for matching
	words = (unsigned int *) malloc(nwords*sizeof(unsigned int));

	for (int w=0; w<nwords; w++)
	{
		words[w] = ((unsigned int) keywords[w][0])
             + ((unsigned int) keywords[w][1])*(1<<8)
             + ((unsigned int) keywords[w][2])*(1<<16)
             + ((unsigned int) keywords[w][3])*(1<<24);

	}

	printf("HEY   %d %d\n",text[1485], text[1486]);

	err=cudaMemcpyToSymbol(wordlist, words, nwords*sizeof(unsigned int));

	err=cudaMalloc((void**)&d_text, len*sizeof(unsigned int));
	err=cudaMalloc((void**)&d_matches, nwords*sizeof(unsigned int));

	err=cudaMemcpy(d_text, text, len*sizeof(unsigned int), cudaMemcpyHostToDevice);
	err=cudaMemcpy(d_matches, matches1, nwords*sizeof(unsigned int), cudaMemcpyHostToDevice);

	//dim3 dimBlock(4,4);
    //dim3 dimGrid(32,16);

	// CPU execution
	const clock_t begin_time = clock();
	matchPattern_CPU(text, words, matches, nwords, len);
	float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	printf("Time for matching keywords: %fs\n\n", runTime);

	printf("CPU Printing Matches:\n");
	printf("Word\t  |\tNumber of Matches\n===================================\n");
	for (int i = 0; i < nwords; ++i)
		printf("%s\t  |\t%d\n", keywords[i], matches[i]);

	// GPU execution
	const clock_t begin_time1 = clock();

	matchPattern_GPU<<<ceil(len/512.0), 512>>>(d_text, d_matches, nwords, len);

    err=cudaMemcpy(matches1, d_matches, nwords*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	float runTime1 = (float)( clock() - begin_time1 ) /  CLOCKS_PER_SEC;
	printf("Time for matching keywords: %fs\n\n", runTime1);

	printf("GPU Printing Matches:\n");
	printf("Word\t  |\tNumber of Matches\n===================================\n");
	for (int i = 0; i < nwords; ++i)
		printf("%s\t  |\t%d\n", keywords[i], matches1[i]);

	free(ctext);
	free(words);
}
