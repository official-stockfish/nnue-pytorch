#pragma once

#if defined (__x86_64__)
#    define EXPORT
#    define CDECL
#elif defined (_MSC_VER)
#    define EXPORT __declspec(dllexport)
#    define CDECL __cdecl
#else
#    define EXPORT
#    define CDECL __attribute__ ((__cdecl__))
#endif
