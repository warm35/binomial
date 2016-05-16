/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <cutil.h>



////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on CPU.
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////
extern "C"
void binomialOptionsCPU(
    float *h_CallResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int OptN
);



////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#define BLOCK_N 64
#include "binomialOptions_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
#ifdef __DEVICE_EMULATION__
//Due to thread synchronization in CUDA kernel
//device emulation runs extremly slow, so reduce problem size for this mode
const int OPT_N = 1;
#else
const int OPT_N = 512;
#endif

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    //'h_' prefix - CPU (host) memory space
    float
    //CPU instance of input data
        *h_StockPrice,
        *h_OptionStrike,
        *h_OptionYears,
    //Results calculated by CPU for reference
        *h_CallResultCPU,
    //CPU copy of GPU results
        *h_CallResultGPU;

    //'d_' prefix - GPU (device) memory space
    float
    //GPU instance of input data
        *d_StockPrice,
        *d_OptionStrike,
        *d_OptionYears,
    //Results calculated by GPU
        *d_CallResult;

    double
        delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    unsigned int hTimer;
    int i;



    CUT_DEVICE_INIT();
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );

    printf("Initializing data...\n");
        printf("...allocating CPU memory.\n");
        h_StockPrice    = (float *)malloc(OPT_SZ);
        h_OptionStrike  = (float *)malloc(OPT_SZ);
        h_OptionYears   = (float *)malloc(OPT_SZ);
        h_CallResultCPU = (float *)malloc(OPT_SZ);
        h_CallResultGPU = (float *)malloc(OPT_SZ);

        printf("...allocating GPU memory.\n");
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_StockPrice,  OPT_SZ) );
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_OptionStrike, OPT_SZ) );
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_OptionYears,  OPT_SZ) );
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_CallResult,   OPT_SZ) );

        printf("...generating input data in CPU mem.\n");
        srand(5347);
        //Generate options set
        for(i = 0; i < OPT_N; i++){
            h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
            h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
            h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
            h_CallResultCPU[i] = -1.0f;
        }

        printf("...copying input data to GPU mem.\n");
        //Copy options data to GPU memory for further processing
        CUDA_SAFE_CALL( cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(d_CallResult,   h_CallResultCPU, OPT_SZ, cudaMemcpyHostToDevice) );
    printf("Data init done.\n");


    printf("Executing GPU kernel...\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        CUT_SAFE_CALL( cutResetTimer(hTimer) );
        CUT_SAFE_CALL( cutStartTimer(hTimer) );
        binomialOptionsGPU<<<BLOCK_N, 256>>>(
            d_CallResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
        CUT_CHECK_ERROR("binomialOptionsGPU() execution failed\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        CUT_SAFE_CALL( cutStopTimer(hTimer) );
        gpuTime = cutGetTimerValue(hTimer);
    printf("Options count            : %i     \n", OPT_N);
    printf("Time steps               : %i     \n", NUM_STEPS);
    printf("binomialOptionsGPU() time: %f msec\n", gpuTime);
    printf("Options per second       : %f     \n", OPT_N / (gpuTime * 0.001));


    printf("Reading back the results...\n");
        CUDA_SAFE_CALL( cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost) );


    printf("Checking the results...\n");
        printf("...running CPU calculations.\n");
        //Calculate options values on CPU
        binomialOptionsCPU(
            h_CallResultCPU,
            h_StockPrice,
            h_OptionStrike,
            h_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );

        printf("...comparing the results.\n");
        //Calculate max absolute difference and L1 distance
        //between CPU and GPU results
        sum_delta = 0;
        sum_ref   = 0;
        max_delta = 0;
        for(i = 0; i < OPT_N; i++){
            delta = abs(h_CallResultCPU[i] - h_CallResultGPU[i]);
            ref   = h_CallResultCPU[i];
            if(delta > max_delta) max_delta = delta;
            sum_delta += delta;
            sum_ref   += ref;
        }
        L1norm = sum_delta / sum_ref;
        printf("L1 norm: %E\n", L1norm);
        printf("Max absolute error: %E\n", max_delta);
    printf((L1norm < 5e-4) ? "TEST PASSED\n" : "***TEST FAILED!!!***\n");


    printf("Shutting down...\n");
        printf("...releasing GPU memory.\n");
        CUDA_SAFE_CALL( cudaFree(d_CallResult)   );
        CUDA_SAFE_CALL( cudaFree(d_OptionYears)  );
        CUDA_SAFE_CALL( cudaFree(d_OptionStrike) );
        CUDA_SAFE_CALL( cudaFree(d_StockPrice)  );

        printf("...releasing CPU memory.\n");
        free(h_CallResultGPU);
        free(h_CallResultCPU);
        free(h_OptionYears);
        free(h_OptionStrike);
        free(h_StockPrice);
        CUT_SAFE_CALL( cutDeleteTimer(hTimer) );
    printf("Shutdown done.\n");

    CUT_EXIT(argc, argv);
}
