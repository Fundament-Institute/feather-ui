// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__LAYOUT_FLEX_H
#define FG__LAYOUT_FLEX_H

#include "../outline.h"

#ifdef  __cplusplus
extern "C" {
#endif

fgDocumentNode* fgLayoutFlex(struct FG__OUTLINE_NODE*, fgDocumentNode*, const fgRect* area, const fgOutlineNode* parent, float scale, fgVec dpi);

#ifdef  __cplusplus
}
#endif

#endif
