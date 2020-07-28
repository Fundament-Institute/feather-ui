// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__UTF_H
#define GL__UTF_H

#include "compiler.h"
#include <stdint.h>

size_t UTF8toUTF32(const char* FG_RESTRICT input, ptrdiff_t srclen, char32_t* FG_RESTRICT output, size_t buflen);

#endif