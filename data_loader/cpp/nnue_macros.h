#ifndef NNUE_MACROS_H
#define NNUE_MACROS_H

/* * 1. Symbol Visibility (NNUE_EXPORT)
 * Ensures functions are exported to the dynamic symbol table of the .so/.dll.
 */
#if defined(_MSC_VER)
    #define NNUE_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
    #define NNUE_EXPORT __attribute__((visibility("default")))
#else
    #define NNUE_EXPORT
#endif

/* * 2. Calling Convention (NNUE_CDECL)
 * The cdecl calling convention is only applicable to 32-bit x86 architectures.
 * 64-bit x86 (x86_64) and ARM (aarch64, etc.) use unified calling conventions,
 * so explicitly defining cdecl there triggers compiler warnings or errors.
 */
#if defined(_M_IX86) || defined(__i386__)
    #if defined(_MSC_VER)
        #define NNUE_CDECL __cdecl
    #else
        #define NNUE_CDECL __attribute__((__cdecl__))
    #endif
#else
    #define NNUE_CDECL
#endif

/* * 3. C-Linkage (Name Mangling Prevention)
 * Forces the C++ compiler to disable name mangling for these functions.
 * This is strictly required for Python's ctypes or other FFI to find the
 * function by its exact string name (e.g., "destroy_fen_batch").
 */
#ifdef __cplusplus
    #define NNUE_EXTERN_C extern "C"
#else
    #define NNUE_EXTERN_C
#endif

/*
 * 4. Combined API Macro
 * A convenience wrapper to apply both C-linkage and visibility at once.
 */
#define NNUE_API NNUE_EXTERN_C NNUE_EXPORT

/*
 * 5. Macros for manual branching optimization for pgo build.
*/

#if defined(__GNUC__) || defined(__clang__)
    #define NNUE_COLD __attribute__((cold))
#else
    #define NNUE_COLD
#endif

#endif // NNUE_MACROS_H