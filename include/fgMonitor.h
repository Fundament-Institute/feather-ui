// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_MONITOR_H__
#define __FG_MONITOR_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Defines a helper element that may OPTIONALLY be used by an implementation to track all available monitors. If a monitor's DPI
// does not match the root DPI, elements must children of that monitor in order to respond to it's DPI.
typedef struct _FG_MONITOR {
  fgElement element;
  AbsRect coverage; // True area of the monitor in the native DPI of root.
  fgIntVec dpi; // DPI of the monitor
  struct _FG_MONITOR* mnext;
  struct _FG_MONITOR* mprev;
} fgMonitor;

struct _FG_ROOT;

FG_EXTERN void FG_FASTCALL fgMonitor_Init(fgMonitor* BSS_RESTRICT self, fgFlag flags, struct _FG_ROOT* BSS_RESTRICT parent, fgMonitor* BSS_RESTRICT prev, const AbsRect* coverage, const fgIntVec* dpi);
FG_EXTERN void FG_FASTCALL fgMonitor_Destroy(fgMonitor* self);
FG_EXTERN size_t FG_FASTCALL fgMonitor_Message(fgMonitor* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif