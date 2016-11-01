// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_GRID_H__
#define __FG_GRID_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

  // Represents a grid of columns (or rows) with labels.
  typedef struct _FG_GRID {
    fgBox box;
    fgBox header; // The header is where the labels are
#ifdef  __cplusplus
    inline operator fgElement*() { return &box.scroll.control.element; }
    inline fgElement* operator->() { return operator fgElement*(); }
    FG_DLLEXPORT void AddColumn(const char* name, unsigned short column = (unsigned short)-1);
    FG_DLLEXPORT void AddItem(fgElement* item, unsigned short column, unsigned short row);
    FG_DLLEXPORT bool RemoveColumn(unsigned int column);
    FG_DLLEXPORT bool RemoveRow(unsigned int row);
    FG_DLLEXPORT bool RemoveItem(unsigned int column, unsigned int row);
    FG_DLLEXPORT fgElement* GetItem(unsigned int column, unsigned int row);
#endif
  } fgGrid;

  FG_EXTERN void FG_FASTCALL fgGrid_Init(fgGrid* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
  FG_EXTERN void FG_FASTCALL fgGrid_Destroy(fgGrid* self);
  FG_EXTERN size_t FG_FASTCALL fgGrid_Message(fgGrid* self, const FG_Msg* msg);

  FG_EXTERN size_t FG_FASTCALL fgGridRow_Init(fgGrid* self, const FG_Msg* msg);
  FG_EXTERN size_t FG_FASTCALL fgGridRow_Message(fgGrid* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
