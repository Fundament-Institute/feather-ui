// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__LAYOUT_DEFAULT_H
#define FG__LAYOUT_DEFAULT_H

#include "../outline.h"

#ifdef  __cplusplus
extern "C" {
#endif

void fgLayoutDefault(fgDocumentNode*, const fgRect* area, const fgOutlineNode* parent, float scale, fgVec dpi);
unsigned char fgLayoutDefaultResolver(void* ptr, const char* id, fgCalcResult* out, int* n);

#ifdef  __cplusplus
}
#endif

#endif
