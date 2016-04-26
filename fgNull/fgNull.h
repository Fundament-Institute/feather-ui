// Copyright ©2016 Black Sphere Studios
// FgNull - A null-implementation of Feather GUI
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_NULL_H__
#define __FG_NULL_H__

#include "fgControl.h"

#ifdef  __cplusplus
extern "C" {
#endif

FG_EXTERN void (FG_FASTCALL *debugmsghook)(fgControl* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
