#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
class test1
{
public:
	test1();
	~test1();
	void Excute();

private:
	void MatrixMultiply(int block_size, const dim3& dimsA, const dim3& dimsB);
};


