// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__WINDOW_H
#define FG__WINDOW_H

#include "../message.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Contains additional outline information for a "box" behavior function
typedef struct
{
  const char* title;
  const void* icon;
} fgWindowData;

enum FG_WINDOW_STATES
{
  FG_WINDOW_TITLE = 0x10,
  FG_WINDOW_ICON,
  FG_WINDOW_RECT,
  FG_WINDOW_STATE,
  FG_WINDOW_QUIET = 0x8000,
};

enum FG_WINDOW_ACTIONS
{
  FG_WINDOW_ONMINIMIZE = 1,
  FG_WINDOW_ONMAXIMIZE,
  FG_WINDOW_ONOPEN,
  FG_WINDOW_ONCLOSE,
};

enum FG_WINDOW_FLAGS
{
  FG_WINDOW_MAXIMIZABLE = (1 << 0),
  FG_WINDOW_MINIMIZABLE = (1 << 1),
  FG_WINDOW_RESIZABLE   = (1 << 2),
  FG_WINDOW_MINIMIZED   = (1 << 3),
  FG_WINDOW_MAXIMIZED   = (1 << 4),
  FG_WINDOW_PREVIEW     = (1 << 5),
  FG_WINDOW_CLOSED      = (1 << 6),
};

typedef struct
{
  fgRect area;
  int flags;
} fgWindowState;

fgMessageResult fgWindowBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);
unsigned char fgWindowResolver(void* data, const char* id, fgCalcResult* out, int* n);

#ifdef  __cplusplus
}
#endif

#endif
