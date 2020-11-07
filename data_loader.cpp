#include <iostream>

#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif

extern "C" {

    EXPORT void CDECL test()
    {
        std::cout<< "test successful\n";
    }

}
