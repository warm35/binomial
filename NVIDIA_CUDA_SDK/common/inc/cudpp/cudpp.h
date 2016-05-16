// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Source: $
// $Revision: 3572$
// $Date: 2007-11-05 09:19:21 +0000 (Mon, 05 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of cudpp_license.txt 
// in the root "docs" directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp.h
 * 
 * @brief Main library header file.  Defines public interface.
 *
 * The CUDPP public interface is a C-only interface to enable 
 * linking with code written in other languages (e.g. C, C++, 
 * and Fortran).  While the internals of CUDPP are not limited 
 * to C (some C++ features are used), the public interface is 
 * entirely C (thus it is declared "extern C").
 */

/**
 * \mainpage
 *
 * \section introduction Introduction
 * 
 * CUDPP is the CUDA Data Parallel Primitives Library. CUDPP is a
 * library of data-parallel algorithm primitives such as 
 * parallel-prefix-sum ("scan"), parallel sorting and parallel 
 * reduction. Primitives such as these are important building blocks 
 * for a wide variety of data-parallel algorithms, including sorting, 
 * stream compaction, and building data structures such as trees and 
 * summed-area tables.
 * 
 * \section homepage Homepage
 * Homepage for CUDPP: http://www.gpgpu.org/developer/cudpp/
 * 
 * Announcements and discussion of CUDPP are hosted on 
 * the <a href="http://groups.google.com/group/cudpp?hl=en">CUDPP Google Group</a>.
 * 
 * \section getting-started Getting Started with CUDPP
 *
 * You may want to start by browsing the \link publicInterface CUDPP
 * public interface\endlink. For information on building CUDPP, see 
 * \ref building-cudpp "Building CUDPP".
 *
 * The "apps" subdirectory included with CUDPP has a few source code samples that 
 * use CUDPP:
 * - \ref example_simpleCUDPP "simpleCUDPP", a simple example of using cudppScan()
 * - satGL, an example of using cudppMultiScan() to generate a summed-area-table of a
 * scene rendered in real time.  The SAT is then used to simulate depth of field.
 * - cudpp_testrig, a comprehensive test application for all the functionality of CUDPP
 *
 * We have also provided a code walkthrough of the \ref example_simpleCUDPP "simpleCUDPP"
 * example.
 *
 * \section release-notes Release Notes
 *
 * \note This release (rel_gems3) should be considered beta code.  This release was made to 
 * support the GPU Gems 3 article "Parallel Prefix Sum (Scan) with CUDA" (see 
 * \ref references "references").  Because this is beta code, the interfaces in the CUDPP 
 * library may change in a future release.  In fact, some of the interfaces <i>will</i> 
 * change (for the better).  To see some discussion about what will change, see the 
 * \ref todo "todo" page.
 *
 * This release (rel_gems3) has been tested on 32-bit Windows XP and 64-bit Redhat 
 * Enterprise Linux 5 (RHEL 5 x86_64).  We expect the code to compile and work 
 * correctly on other linux flavors, but it has not yet been tested.
 *
 * \section cuda CUDA
 * CUDPP is implemented in <a href="http://developer.nvidia.com/cuda">NVIDIA CUDA</a>.
 * It requires the CUDA Toolkit version 1.0 or later.  Please see the NVIDIA 
 * <a href="http://developer.nvidia.com/cuda">CUDA</a> homepage to download CUDA as well
 * as the CUDA Programming Guide and CUDA SDK, which includes many CUDA code examples.
 *
 * \section design-goals Design Goals
 * Design goals for CUDPP include:
 * 
 * - Performance. We aim to provide best-of-class performance for our
 *   primitives. We welcome suggestions and contributions that will
 *   improve our performance. We also want to provide primitives that
 *   can be easily benchmarked, and compared against other
 *   implementations on GPUs and other processors.
 * - Modularity. We want our primitives to be easily included in other
 *   applications. To that end we have made the following design decisions:
 *   - CUDPP is provided as a library that can link against other applications. 
 *   - CUDPP calls run on the GPU on GPU data. Thus they can be used
 *     as standalone calls on the GPU (with programmers initializing
 *     GPU data and performing copies back and forth) and, more
 *     importantly, as GPU components in larger CPU/GPU applications.
 *   - CUDPP is implemented as 4 layers:
 *     -# The \link publicInterface Public Interface\endlink is the external 
 *        library interface, which is the intended entry point for most applications.
 *        The public interface calls into the \link cudpp_app Application-Level API\endlink.
 *     -# The \link cudpp_app Application-Level API\endlink comprises functions
 *        callable from CPU code.  These functions execute code jointly on the CPU 
 *        and the GPU by calling into the \link cudpp_kernel Kernel-Level API\endlink
 *        below them.
 *     -# The \link cudpp_kernel Kernel-Level API\endlink comprises functions
 *        that run entirely on the GPU across an entire grid of thread blocks.  These
 *        functions may call into the \link cudpp_cta CTA-Level API\endlink below them.
 *     -# The \link cudpp_cta CTA-Level API\endlink  comprises functions that run 
 *        entirely on the GPU within a single Cooperative Thread Array (CTA, aka Thread
 *        block).  These are low-level functions implementing core data-parallel 
 *        algorithms, typically by processing data within shared (CUDA \c __shared__)
 *        memory.
 *
 * Programmers may use any of the lower three CUDPP layers in their own programs by building 
 * the source directly into their application.  However, the typical usage of CUDPP is 
 * to link to the library and invoke functions in the CUDPP 
 * \link publicInterface Public Interface\endlink, as in the \ref example_simpleCUDPP 
 * "simpleCUDPP", satGL, and cudpp_testrig application examples included in the CUDPP
 * distribution.
 *
 * In the future, if and when CUDA supports building device-level libraries, we hope to 
 * enhance CUDPP to ease the use of CUDPP internal algorithms at all levels.
 *
 * \subsection uses Use Cases
 * We expect the normal use of CUDPP will be in one of two ways:
 * -# Linking the CUDPP library against another application. 
 * -# Running our "test" application, cudpp_testrig, that exercises
 *   CUDPP functionality.
 *
 * \section references References
 * The following publications describe work incorporated in CUDPP.
 * 
 * - Mark Harris, Shubhabrata Sengupta, and John D. Owens. "Parallel Prefix Sum (Scan) with CUDA". In Hubert Nguyen, editor, <i>GPU Gems 3</i>, chapter 39, pages 851&ndash;876. Addison Wesley, August 2007. http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=916
 * - Shubhabrata Sengupta, Mark Harris, Yao Zhang, and John D. Owens. "Scan Primitives for GPU Computing". In <i>Graphics Hardware 2007</i>, pages 97&ndash;106, August 2007. http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=915
 *
 * \section credits Credits
 * \subsection developers CUDPP Developers
 * - <a href="http://www.markmark.net">Mark Harris</a>, NVIDIA Ltd.
 * - <a href="http://www.ece.ucdavis.edu/~jowens/">John D. Owens</a>, University of California, Davis
 * - Shubho Sengupta, University of California, Davis
 * - Yao Zhang,       University of California, Davis
 * - Andrew Davidson, Louisiana State University
 * 
 * \subsection contributors Other CUDPP Contributors
 * - Nadatur Satish,  University of California, Berkeley
 *
 * \subsection acknowledgments Acknowledgments
 *
 * Thanks to Jim Ahrens, Ian Buck, Guy Blelloch, Jeff Bolz, Jeff
 * Inman, Eric Lengyel, David Luebke, Pat McCormick, and Richard
 * Vuduc for their contributions during the development of this library. 
 * 
 * Thanks also to our funding agencies:
 * - Department of Energy Early Career Principal Investigator Award
 *   DE-FG02-04ER25609
 * - SciDAC Institute for Ultrascale Visualization (http://www.iusv.org/)
 * - Los Alamos National Laboratory
 * - National Science Foundation (grant 0541448)
 * - Generous hardware donations from NVIDIA
 *
 * \section license-overview CUDPP Copyright and Software License
 * CUDPP is copyright The Regents of the University of California, Davis campus and
 * NVIDIA Corporation.  The license is a modified version of the BSD license, designed
 * to encourage reuse of this software in other projects, both commercial and 
 * non-commercial.  For details, please see the \ref license page. 
 */

/**
 * @page license CUDPP License
 *
 * @include license.txt
 */

/**
 * @page building-cudpp Building CUDPP
 *
 * CUDPP has currently been tested in 32-bit Windows XP and Linux.  See \ref release-notes
 * for release specific platform support.
 *
 * \section build-win32 Building CUDPP on Windows XP
 *
 * CUDPP can be built using either MSVC 7.1 (.NET 2003) or MSVC 8 (2005).  To build, open either
 * cudpp/cudpp.sln or cudpp_vc7.sln, depending on whether you have MSVC 8 or MSVC 7, respectively.
 * Then you can build the library using the "build" command as you would with any other workspace.
 * There are four configurations: debug, release, emudebug, and emurelease.  The first two are 
 * self-explanatory.  The second two are built to use CUDA device emulation, meaning they will
 * be run (slowly) on the CPU.
 *
 * \section build-linux Building CUDPP on Linux
 *
 * CUDPP can be built using standard g++ and Make tools on Linux, by typing "make" in the "cudpp/"
 * subdirectory.  Before building CUDPP, you should first build the CUDA Utility Library (libcutil)
 * by typing "make; make dbg=1" in the "common/" subdirectory.  This will generate libcutil.a and
 * libcutilD.a.  
 * 
 * The makefile for CUDPP and all sample applications take the optional arguments
 * "emu=1" and "dbg=1".  The former builds CUDPP for device emulation, and the latter for debugging.
 * The two flags can be combined.
 *
 * \section build-apps Building CUDPP Sample Applications
 * 
 * The sample applications in the "apps/" subdirectory can be built exactly like CUDPP is--either by
 * opening the appropriate .sln/.vcproj file in MSVC in Windows, or using "make" in Linux.
 * 
 */

#ifndef __CUDPP_H__
#define __CUDPP_H__

#include <stdlib.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * Options for configuring scan.
 * 
 * @see CUDPPScanConfig, cudppScan
 */
enum CUDPPScanOption
{
    CUDPP_SCAN_FORWARD,   /**< Forward scan - scans from start to end of
                           * array */
    CUDPP_SCAN_BACKWARD,  /**< Backward scan - scans from end to start
                           * of array */
    CUDPP_SCAN_EXCLUSIVE, /**< Exclusive scan - scan includes all
                           * elements up to (but not including) the
                           * current element */
    CUDPP_SCAN_INCLUSIVE, /**< Inclusive scan - scan includes all
                           * elements up to and including the current
                           * element */
};

/** 
 * Datatype enum (used in various CUDPP routines).
 *
 * @see CUDPPScanConfig, cudppScan
 */
enum CUDPPDatatype
{
    CUDPP_CHAR,
    CUDPP_UCHAR,
    CUDPP_INT, 
    CUDPP_UINT, 
    CUDPP_FLOAT
};

/** 
 * Operator enum (which binary associative operator is used for scan).
 *
 * @see CUDPPScanConfig, cudppScan
 */
enum CUDPPOperator
{
    CUDPP_ADD,
    CUDPP_MULTIPLY,
    CUDPP_MIN,
    CUDPP_MAX
};

/**
 * @brief Configuration information for scan calls.
 * 
 * @todo [MJH] The scan (and all other) interfaces will be replaced with an 
 * FFTW-style "plan" interface.  Scan plans will be referenced by handle, 
 * for example.
 */
struct CUDPPScanConfig
{
    // parameters
    unsigned int  direction;          //!< Forward or backward scan
    unsigned int  exclusivity;        //!< Exclusive or inclusive scan (see cudppScan())
    CUDPPOperator op;                 //!< Binary associative operator
    CUDPPDatatype datatype;           //!< Datatype to be scanned

    unsigned int  maxNumElements;     //!< Maximum size of scan (for allocation)
    unsigned int  maxNumRows;         //!< Number of simultaneous scans (for cudppMultiScan())

    size_t		  rowPitch;           //!< The width in elements of each row (for cudppMultiScan())

    // system only
    void **_scanBlockSums;            //!< @internal Intermediate block sums array
    unsigned int *_rowPitches;        //!< @internal Pitch of each row in elements (for cudppMultiScan())
    unsigned int _numEltsAllocated;   //!< @internal Number of elements allocated (maximum scan size)
    unsigned int _numRowsAllocated;   //!< @internal Number of rows allocated (for cudppMultiScan())
    unsigned int _numLevelsAllocated; //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/**
 * For sorts, which sort algorithm will be used? 
 * 
 * @see CUDPPSortConfig, cudppSort
 */
enum CUDPPSortAlgorithm
{
    CUDPP_SORT_RADIX,        /**< Radix sort within chunks, merge sort to
                              * merge chunks together */
    CUDPP_SORT_RADIX_GLOBAL, /**< Global radix sort across entire
                              * input, no merge */
    CUDPP_SORT_INVALID,      /**< Placeholder at end of enum */
};

/**
 * @brief Configuration information for sort calls.
 * 
 * @param datatype Which datatype is being sorted (currently only int
 * and float work)
 * @param sortAlgorithm Which sort algorithm will be used to sort?
 * @param numElements How many elements will be sorted?
 * @param scanConfig Since scan is used in many sorts, scan must be
 * configured here.
 * 
 * @see cudppSort, cudppScan, CUDPPDatatype, CUDPPSortAlgorithm, CUDPPScanConfig
 */
struct CUDPPSortConfig
{
    CUDPPDatatype       datatype;
    CUDPPSortAlgorithm  sortAlgorithm;

    unsigned int        numElements;
    CUDPPScanConfig    *scanConfig;
};

#ifndef CUDPP_DLL
#ifdef _WIN32
#ifdef BUILD_DLL
#define CUDPP_DLL __declspec(dllexport)
#else
#define CUDPP_DLL __declspec(dllimport)
#endif
#else
#define CUDPP_DLL
#endif
#endif

CUDPP_DLL
void cudppInitializeScan(CUDPPScanConfig *config);

CUDPP_DLL
void cudppFinalizeScan(CUDPPScanConfig *config);

CUDPP_DLL
void cudppScan(void *deviceOut, 
               const void *deviceIn, 
               int numElements,
               CUDPPScanConfig *config);

CUDPP_DLL
void cudppMultiScan(void *deviceOut, 
                    const void *deviceIn, 
                    int numElements,
                    int numRows,
                    CUDPPScanConfig *config);

CUDPP_DLL
void cudppCompact(void *deviceOut, 
                  unsigned int *deviceIsValid,
                  const void *deviceIn, 
                  int numElements,
                  CUDPPScanConfig *config,
                  unsigned int* numValidElements);

CUDPP_DLL
void cudppSort(void *deviceOut, 
               const void *deviceIn, 
               void *deviceTemp, 
               void *deviceTemp2, 
               const CUDPPSortConfig * config,
               bool justChunks);

#ifdef __cplusplus
}
#endif

#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
