/*

Copyright 2020 Tomasz Sobczyk

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#pragma once

#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <climits>
#include <ctime>
#include <cstdint>



namespace binpack
{
    [[nodiscard]] inline std::uint16_t signedToUnsigned(std::int16_t a)
    {
        std::uint16_t r;
        std::memcpy(&r, &a, sizeof(std::uint16_t));
        if (r & 0x8000)
        {
            r ^= 0x7FFF;
        }
        r = (r << 1) | (r >> 15);
        return r;
    }

    [[nodiscard]] inline std::int16_t unsignedToSigned(std::uint16_t r)
    {
        std::int16_t a;
        r = (r << 15) | (r >> 1);
        if (r & 0x8000)
        {
            r ^= 0x7FFF;
        }
        std::memcpy(&a, &r, sizeof(std::uint16_t));
        return a;
    }
}