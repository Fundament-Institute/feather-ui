// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TOOLBAR_H__
#define __FG_TOOLBAR_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTOOLBAR_FLAGS
{
  FGTOOLBAR_LOCKED = (FGCONTROL_DISABLE << 1), // Locks the toolgroups so they can't be moved or re-arranged.
};

// A toolbar is simply a list of lists containing buttons or other controls.
typedef struct _FG_TOOLBAR {
  fgBox box; // ADDITEM adds a toolgroup. ADDITEM on a toolgroup adds an fgBox, to which you then add buttons
#ifdef  __cplusplus
  inline operator fgElement*() { return &box.scroll.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgToolbar;

FG_EXTERN void fgToolbar_Init(fgToolbar* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
FG_EXTERN void fgToolbar_Destroy(fgToolbar* self);
FG_EXTERN size_t fgToolbar_Message(fgToolbar* self, const FG_Msg* msg);

FG_EXTERN void fgToolGroup_Init(fgBox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
FG_EXTERN size_t fgToolGroup_Message(fgBox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif