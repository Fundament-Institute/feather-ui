// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__TEXT_H
#define FG__TEXT_H

#include "../message.h"
#include "../backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Contains additional outline information for a "box" behavior function
typedef struct
{
  const char* family;
  unsigned short weight;
  bool italic;
  float letterSpacing;
  const char* text;
  fgColor color;
  float blur;
  FG_TEXT_ANTIALIASING aa;
} fgTextData;

typedef struct
{
  fgFont* font;
  fgFontLayout layout;
} fgTextState;

FG_COMPILER_DLLEXPORT fgMessageResult fgTextBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);

#ifdef  __cplusplus
}
#endif

#endif