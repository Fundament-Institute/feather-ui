// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__LAYOUT_DEFAULT_H
#define FG__LAYOUT_DEFAULT_H

#include "../outline.h"

#ifdef  __cplusplus
extern "C" {
#endif

FG_COMPILER_DLLEXPORT void fgLayoutDefault(fgDocumentNode*, const fgRect* area, const fgOutlineNode* parent, float scale, fgVec dpi);
FG_COMPILER_DLLEXPORT unsigned int fgLayoutDefaultResolver(void* outline, unsigned int index, fgCalcNode* out, const char* id);

#ifdef  __cplusplus
}
#endif

#endif
