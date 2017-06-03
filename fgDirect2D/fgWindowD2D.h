// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_WINDOW_D2D_H__
#define __FG_WINDOW_D2D_H__

#include "fgWindow.h"
#include "fgContext.h"
#include "bss-util/LLBase.h"
#include <stack>

struct fgWindowD2D {
  fgWindow window;
  fgContext context;
  HWND__* handle;
  AbsVec dpi;
  bss::LLBase<fgWindowD2D> list;

  static longptr_t __stdcall WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);
  static fgWindowD2D* windowlist;
  static BSS_FORCEINLINE bss::LLBase<fgWindowD2D>& GETNODE(fgWindowD2D* n) { return n->list; }
};

#ifdef  __cplusplus
extern "C" {
#endif

extern void fgWindowD2D_Init(fgWindowD2D* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
extern void fgWindowD2D_Destroy(fgWindowD2D* self);
extern size_t fgWindowD2D_Message(fgWindowD2D* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif