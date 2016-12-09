// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_GRID_H__
#define __FG_GRID_H__

#include "fgList.h"

#ifdef  __cplusplus
extern "C" {
#endif

  typedef struct _FG_GRID_ROW {
    fgBox box;
#ifdef  __cplusplus
    inline operator fgElement*() { return &box.scroll.control.element; }
    inline fgElement* operator->() { return operator fgElement*(); }
    FG_DLLEXPORT void InsertItem(fgElement* item, size_t column = (size_t)-1);
    FG_DLLEXPORT void InsertItem(const char* item, size_t column = (size_t)-1);
    FG_DLLEXPORT bool RemoveItem(size_t column);
    FG_DLLEXPORT fgElement* GetItem(size_t column);
#endif
  } fgGridRow;

  // Represents a grid of columns (or rows) with labels.
  typedef struct _FG_GRID {
    fgBox box;
    fgList header; // The header is where the labels are
    fgVectorElement selected;
    fgColor select; // color index 0
    fgColor hover; // color index 1
#ifdef  __cplusplus
    inline operator fgElement*() { return &box.scroll.control.element; }
    inline fgElement* operator->() { return operator fgElement*(); }
    FG_DLLEXPORT void InsertColumn(const char* name, size_t column = (size_t)-1);
    FG_DLLEXPORT void SetItem(fgElement* item, size_t column, size_t row);
    FG_DLLEXPORT void SetItem(const char* item, size_t column, size_t row);
    FG_DLLEXPORT fgGridRow* InsertRow(size_t row = (size_t)-1);
    FG_DLLEXPORT bool RemoveColumn(size_t column);
    FG_DLLEXPORT bool RemoveRow(size_t row);
    FG_DLLEXPORT bool RemoveItem(size_t column, size_t row);
    FG_DLLEXPORT fgElement* GetItem(size_t column, size_t row);
    FG_DLLEXPORT fgGridRow* GetRow(size_t row);
#endif
  } fgGrid;

  FG_EXTERN void FG_FASTCALL fgGrid_Init(fgGrid* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
  FG_EXTERN void FG_FASTCALL fgGrid_Destroy(fgGrid* self);
  FG_EXTERN size_t FG_FASTCALL fgGrid_Message(fgGrid* self, const FG_Msg* msg);

  FG_EXTERN void FG_FASTCALL fgGridRow_Init(fgGridRow* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
  FG_EXTERN size_t FG_FASTCALL fgGridRow_Message(fgGridRow* self, const FG_Msg* msg);
  FG_EXTERN void FG_FASTCALL fgGridRow_Destroy(fgGridRow* self);

#ifdef  __cplusplus
}
#endif

#endif
