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



////////////////////////////////////////////////////////////////////////////////
// Parameters restriction:
// 1) CACHE_DELTA must be even as we unroll two iterations
// 2) NUM_STEPS and CACHE_STEP must be multiples of CACHE_DELTA.
//    Though current implemenation can be modified to accept
//    arbitrary cache parameters, "bad" (not multiples of warp size)
//    CACHE_STEP and NUM_STEPS values can break alignment constraints of
//    memory coalescing, thus hitting overall memory performance
////////////////////////////////////////////////////////////////////////////////
#define  CACHE_STEP 512
#define CACHE_DELTA 32
#define  CACHE_SIZE (CACHE_STEP + CACHE_DELTA)



//Number of time steps
#define NUM_STEPS 2048

//End prices storage for each thread block
//(NUM_STEPS + 1) node count is aligned to a multiple of 16 for coalescing
static __device__ float steps[BLOCK_N * (NUM_STEPS + 16)];


////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void binomialOptionsGPU(
    float *d_CallResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int OptN
){
    //Prices cache
    __shared__ float dataA[CACHE_SIZE];
    __shared__ float dataB[CACHE_SIZE];
    //End prices array for current thread block
    float *p = &steps[blockIdx.x * (NUM_STEPS + 16)];

    for(int opt = blockIdx.x; opt < OptN; opt += gridDim.x){
        const float S = d_StockPrice[opt];
        const float X = d_OptionStrike[opt];
        const float T = d_OptionYears[opt];

        //Time per step
        const float dT    = T * (1.0f / NUM_STEPS);
        const float VsdT  = Volatility * sqrtf(dT);
        const float RdT   = Riskfree * dT;
        //Per-step cont. compounded riskless rate
        const float R     = expf(RdT);
        const float Rinv  = 1.0f / R;
        //Up move corresponding to given variance
        const float u     = expf(VsdT);
        //Corresponding down move
        const float d     = 1.0f / u;
        //Pseudo-probability of upward move
        const float Pu    = (R - d) / (u - d);
        //Pseudo-probability of downard move
        const float Pd    = 1.0f - Pu;
        //Compounded quotents
        const float PuByR = Pu * Rinv;
        const float PdByR = Pd * Rinv;


        ///////////////////////////////////////////////////////////////////////
        // Compute values at expiration date:
        // call option value at period end is V(T) = S(T) - X
        // if S(T) is greater than X, or zero otherwise.
        // The computation is similar for put options.
        ///////////////////////////////////////////////////////////////////////
        for(int i = threadIdx.x; i <= NUM_STEPS; i += blockDim.x){
            float price = S * expf(VsdT * (2.0f * i - NUM_STEPS));
            p[i]        = fmaxf(price - X, 0);
        }

        ////////////////////////////////////////////////////////////////////////
        // Walk backwards up binomial tree.
        // Can't do in-place reduction, since warps are scheduled out of order.
        // So double-buffer and synchronize to avoid read-after-write hazards.
        ////////////////////////////////////////////////////////////////////////
        for(int i = NUM_STEPS; i > 0; i -= CACHE_DELTA)
            for(int c_base = 0; c_base < i; c_base += CACHE_STEP){
                //Start and end positions within shared memory cache
                int c_start = min(CACHE_SIZE - 1, i - c_base);
                int c_end   = c_start - CACHE_DELTA;

                //Read data(with apron) to shared memory cache
                __syncthreads();
                for(int k = threadIdx.x; k <= c_start; k += blockDim.x)
                    dataA[k] = p[c_base + k];

                //Calculations within shared memory
                for(int k = c_start - 1; k >= c_end;){
                    //Compute discounted expected value
                    __syncthreads();
                    for(int l = threadIdx.x; l <= k; l += blockDim.x)
                        dataB[l] = PuByR * dataA[l + 1] + PdByR * dataA[l];
                    k--;

                    //Compute discounted expected value
                    __syncthreads();
                    for(int l = threadIdx.x; l <= k; l += blockDim.x)
                        dataA[l] = PuByR * dataB[l + 1] + PdByR * dataB[l];
                    k--;
                }

                //Flush shared memory cache
                __syncthreads();
                for(int k = threadIdx.x; k <= c_end; k += blockDim.x)
                    p[c_base + k] = dataA[k];
            }

        //Write the value at the top of the tree to destination buffer
        if(threadIdx.x == 0) d_CallResult[opt] = dataA[0];
    }
}
