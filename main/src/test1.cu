#include"../head/test1.cuh"
#include<cmath>
#include <stdio.h>

test1::test1()
{
}

test1::~test1()
{
}

//////////////////////////////////
template<int BLOCK_SIZE>
__global__ void MatrixMul(float* C, float* A, float* B, int wA, int wB)
{
	// Block index 
	int bx     = blockIdx.x;
	int by     = blockIdx.y;

	// Thread index 
	int tx     = threadIdx.x;
	int ty     = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block ...
	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd   = aBegin + wA - 1;
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block ...
	int bBegin = BLOCK_SIZE * bx;
	int bStep  = wB * BLOCK_SIZE;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread ...
	float Csub = 0;
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		// declaration of the shared memory array As used to
		// store the sub_matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		__syncthreads();
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}
	int cIndex = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx;
	C[cIndex] = Csub;
}



void test1::MatrixMultiply(int block_size, const dim3& dimsA, const dim3& dimsB)
{
	auto ConstantInit = [=](float* vec, const int & vecLen, const float& val)->void
	{
		for (int i = 0; i < vecLen; i++)
			vec[i] = val;
	};

	// Allocate host memory for matrices A and B ...
	unsigned int size_A     = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A;
	cudaMallocHost((void**)&h_A, mem_size_A);
	
	unsigned int size_B     = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B;
	cudaMallocHost((void**)&h_B, mem_size_B);

	// matrix initiation ...
	const float val_A = 1.f;
	ConstantInit(h_A, size_A, val_A);
	const float val_B = .01f;
	ConstantInit(h_B, size_B, val_B);

	// Allocate host matrix C 
	dim3 dimsC( dimsB.x, dimsA.y, 1);
	unsigned int size_C = dimsC.x * dimsC.y;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C;
	cudaMallocHost((void**)&h_C, mem_size_C);

	// Allocate device memory ...
	float* d_A, * d_B, * d_C;
	cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A);
	cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B);
	cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C);

	// copy host memory to device memory ...
	cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
	//cudaMemcpyAsync(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);

	// setup execution parameters ...
	dim3 threads(block_size, block_size, 1);
	dim3 blocks(dimsB.x / threads.x, dimsA.y / threads.y);

	MatrixMul<32>
		<<<blocks, threads, 0 >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);

	//int nIter = 300;
	//for (int j = 0; j < nIter; j++)
	//{
	//	MatrixMul<32>
	//		<<<blocks, threads, 0 >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
	//}

	//cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 12; i++)
		printf("%.8f\n", h_C[i]);

	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-6;  // machine zero

	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
		const auto curVal = h_C[i];
		double abs_err = fabs(h_C[i] - (dimsA.x * val_B));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > eps) {
			printf("Matrix[%05d]=%.8f, ref=%.8f error term is > %E  \n", i,
				h_C[i], dimsA.x * val_B, eps);
			correct = false;
		}
		//else
		//{
		//	printf("Matrix[%05d]=%.8f, ref=%.8f \n", i,
		//		h_C[i], dimsA.x * val_B);
		//}
	}
	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
	// Clean up memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void test1::Excute()
{
	int block_size = 32;

	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
	MatrixMultiply(block_size, dimsA, dimsB);
}
