#include <stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

#define TILE_WIDTH 16

void printinfo()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) { 
		cudaDeviceProp deviceProp; 
		cudaGetDeviceProperties(&deviceProp, device); 
		printf("Device %d is %s\n", device,deviceProp.name);
		printf("compute capability %d.%d.\n",deviceProp.major, deviceProp.minor); 
		printf("total Global Memory %dMiB\n",deviceProp.totalGlobalMem/1024/1024);
		printf("warp size %d\n", deviceProp.warpSize);
		printf("\n");

		printf("clockRate %dGHz\n", deviceProp.clockRate/1024/1024);
		printf("multiProcessorCount %d\n", deviceProp.multiProcessorCount);
		printf("totalConstMem(65536) %d\n", deviceProp.totalConstMem);
		printf("\n");

		printf("maxThreadsDim %d %d %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("maxGridSize %d %d %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("\n");

		printf("sharedMemPerBlock %d\n", deviceProp.sharedMemPerBlock);
		printf("regsPerBlock %d\n", deviceProp.regsPerBlock);
		printf("memPitch %d\n", deviceProp.memPitch);
		printf("maxThreadsPerBlock %d\n", deviceProp.maxThreadsPerBlock);
		printf("\n");
/*
		
		
		
		
		
		printf(" %d\n", deviceProp.);
		printf(" %d\n", deviceProp.);
		printf(" %d\n", deviceProp.);
		printf(" %d\n", deviceProp.);
		printf(" %d\n", deviceProp.);
		printf(" %d\n", deviceProp.);

		
		
		textureAlignment	512	unsigned int
		deviceOverlap	1	int
		kernelExecTimeoutEnabled	1	int
		integrated	0	int
		canMapHostMemory	1	int
		computeMode	0	int
		maxTexture1D	65536	int
+		maxTexture2D	0x0022fc94	int [2]
+		maxTexture3D	0x0022fc9c	int [3]
+		maxTexture1DLayered	0x0022fca8	int [2]
+		maxTexture2DLayered	0x0022fcb0	int [3]
		surfaceAlignment	512	unsigned int
		concurrentKernels	1	int
		ECCEnabled	0	int
		pciBusID	1	int
		pciDeviceID	0	int
		pciDomainID	0	int
		tccDriver	0	int
		asyncEngineCount	2	int
		unifiedAddressing	0	int
		l2CacheSize	393216	int
		maxThreadsPerMultiProcessor	1536	int

*/
		printf("memoryBusWidth %d\n", deviceProp.memoryBusWidth);
		printf("memoryClockRate %dGHz\n", deviceProp.memoryClockRate/1024/1024);




		printf("\n-----------------------------------------------------\n");
	}
}

void MatrixMultiplicationInHost(float* M,float* N,float* P,int Width)
{
	int i,j,k,sum;
	for(i=0;i<Width;i++){
		for(j=0;j<Width;j++){
			sum=0;
			for(k=0;k<Width;k++){
				sum+=M[i*Width+k]*N[k*Width+j];
			}
			P[i*Width+j]=sum;
		}
	}
}

__global__ void MatrixMulKernel(float* Md,float* Nd,float* Pd,int Width)
{
	int Row=blockIdx.y*TILE_WIDTH+threadIdx.y;
	int Col=blockIdx.x*TILE_WIDTH+threadIdx.x;

	float Pvalue=0;

	for(int k=0;k<Width;++k){
		Pvalue+=Md[Row*Width+k]*Nd[k*Width+Col];
	}
	Pd[Row*Width+Col]=Pvalue;
}

void MatrixMultiplication(float* M,float* N,float* P,int Width)
{
	int size=Width*Width*sizeof(float);
	float* Md;
	float* Nd;
	float* Pd;
	cudaMalloc((void**)&Md,size);
	cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd,size);
	cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd,size);

	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
	dim3 dimGrid(Width/TILE_WIDTH,Width/TILE_WIDTH);
	MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,Width);


	cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);
}

void MakeMatrix(float* M,float* N,int Width)
{
	int i,j;
	for(i=0;i<Width;i++){
		for(j=0;j<Width;j++){
			M[i*Width+j]=rand()/1000;
			N[i*Width+j]=rand()/1000;
		}
	}
}

void PrintMatrix(float* M,int Width)
{
	int i,j;
	for(i=0;i<Width;i++){
		for(j=0;j<Width;j++){
			printf("%.0f ",M[i*Width+j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(void)
{
	int n;
	float *M,*N,*P;
	unsigned int timer_compute=0;
	clock_t before,after;
   double  result1,result2;
   
	printinfo();
	srand(time(NULL));
	
	printf("input N(4,8,12,16,20,24...):");
//	freopen("output.txt","w",stdout);
	
	scanf("%d",&n);
//	
	M=(float*)malloc(n*n*sizeof(float));
	N=(float*)malloc(n*n*sizeof(float));
	P=(float*)malloc(n*n*sizeof(float));
	MakeMatrix(M,N,n);

	before  = clock();
	MatrixMultiplication(M,N,P,n);
	result1 = (double)(clock() - before) / CLOCKS_PER_SEC;

   printf("걸린시간은 %5.2f 입니다.\n", result1); 
//	PrintMatrix(M,n);
//	PrintMatrix(N,n);
//	PrintMatrix(P,n);
	before  = clock();
//	MatrixMultiplicationInHost(M,N,P,n);
	result2 = (double)(clock() - before) / CLOCKS_PER_SEC;

   printf("걸린시간은 %5.2f 입니다.\n%f배\n", result2,result2/result1);

//	kernel<<<1,1>>>();
//	PrintMatrix(M,n);
//	PrintMatrix(N,n);
//	PrintMatrix(P,n);
	return 0;
}