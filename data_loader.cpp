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

    struct TestDataCollection
    {
        int size;
        int* data;
    };

    EXPORT TestDataCollection* CDECL create_data_collection()
    {
        return new TestDataCollection{ 10, new int[10]{} };
    }

    EXPORT void CDECL destroy_data_collection(TestDataCollection* ptr)
    {
        delete ptr->data;
        delete ptr;
    }

}
