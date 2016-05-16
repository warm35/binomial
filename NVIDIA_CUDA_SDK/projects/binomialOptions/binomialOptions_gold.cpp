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



#include <math.h>


//Number of time steps
#define NUM_STEPS  2048



static float maxf(float a, float b){
    return (a > b) ? a : b;
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on CPU
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
){
    static float steps[NUM_STEPS + 1];

    for(int opt = 0; opt < OptN; opt++){
        const float S = h_StockPrice[opt];
        const float X = h_OptionStrike[opt];
        const float T = h_OptionYears[opt];

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
        //Pseudo-probabi lity of downard move
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
        for(int i = 0; i <= NUM_STEPS; i++){
            float price = S * expf(VsdT * (2.0f * i - NUM_STEPS));
            steps[i]  = maxf(price - X, 0.0f);
        }


        ////////////////////////////////////////////////////////////////////////
        // Walk backwards up binomial tree
        ////////////////////////////////////////////////////////////////////////
        for(int i = NUM_STEPS; i > 0; i--)
            for(int j = 0; j <= i - 1; j++)
                steps[j] = PuByR * steps[j + 1] + PdByR * steps[j];

        h_CallResult[opt] = steps[0];
    }
}
