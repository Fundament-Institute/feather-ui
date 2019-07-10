// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__LAYER_H
#define FG__LAYER_H

#include "../message.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum LAYER_MOUSE
{
  LAYER_MOUSE_LAYOUT = 1,
  LAYER_MOUSE_TRANSFORM = 2,
  LAYER_MOUSE_BOTH = 3,
};

// Contains additional outline information for a "layer" behavior function
typedef struct
{
  float transform[4][4];
  float opacity;
  enum LAYER_MOUSE mouse;
} fgLayerData;

fgMessageResult fgLayerBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);

#ifdef  __cplusplus
}
#endif

#endif