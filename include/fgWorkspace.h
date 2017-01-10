// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_WORKSPACE_H__
#define __FG_WORKSPACE_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGWORKSPACE_FLAGS
{
  FGWORKSPACE_RULERX = (FGSCROLLBAR_SHOWV << 1), // Show the vertical or horizontal ruler
  FGWORKSPACE_RULERY = (FGWORKSPACE_RULERX << 1),
  FGWORKSPACE_GRIDX = (FGWORKSPACE_RULERY << 1), // Show grid lines on the x or y axis
  FGWORKSPACE_GRIDY = (FGWORKSPACE_GRIDX << 1),
  FGWORKSPACE_SNAPTOX = (FGWORKSPACE_GRIDY << 1), // Uses a layout that snaps all elements to the x or y axis grid.
  FGWORKSPACE_SNAPTOY = (FGWORKSPACE_SNAPTOX << 1),
  FGWORKSPACE_CROSSHAIR = (FGWORKSPACE_SNAPTOY << 1), // Show the crosshairs
};

// A workspace is intended for things like timelines or editors, and can render rulers or grids on either axis.
typedef struct _FG_WORKSPACE {
  fgScrollbar scrollbar;
  fgElement rulers[2];
  fgElement cursors[2]; // cursors that follow the mouse around on the rulers. Invisible unless a skin puts something inside them.
  unsigned int gridcolor[2]; // color of the grid's x and y axis lines.
  unsigned int rulercolor[2]; // color of the major and minor lines on the ruler
  unsigned int crosshaircolor; // color of the lines drawn to where the cursor is
  fgIntVec dpi;
  AbsVec gridsize;
#ifdef  __cplusplus
  inline operator fgElement*() { return &scrollbar.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgWorkspace;

FG_EXTERN void fgWorkspace_Init(fgWorkspace* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgWorkspace_Destroy(fgWorkspace* self);
FG_EXTERN size_t fgWorkspace_Message(fgWorkspace* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif