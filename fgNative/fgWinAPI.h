// Copyright ©2012 Black Sphere Studios
// fgWinAPI - A native WinAPI implementation of Feather.
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_WINAPI_H__
#define __FG_WINAPI_H__

#include "fgRoot.h"
#include "fgButton.h"
#include "fgMenu.h"
#include "fgWindow.h"
#include "fgTabcontrol.h"
#include "fgList.h"
#include "fgCombobox.h"
#include "fgTextbox.h"

#pragma comment(lib,"comctl32.lib")
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#ifdef  __cplusplus
extern "C" {
#endif
  typedef struct {
    fgAsset render;
    void* handle;
  } WinAPIfgAsset;

  typedef struct {
    fgWindow window;
    void* handle;
    ptrdiff_t (BSS_COMPILER_STDCALL *DefWndProc)(void*, unsigned int, size_t, ptrdiff_t);
  } WinAPIfgWindow;

  typedef struct {
    WinAPIfgWindow wn;
    fgWindow region;
    int margin[4]; // ltrb rectangle
  } WinAPIfgTop;

  // WndProc message translation function used by all windows
  FG_EXTERN ptrdiff_t BSS_COMPILER_STDCALL fgWindowWndProc(void* hWnd, unsigned int message, size_t wParam, ptrdiff_t lParam);
  // WndProc message translation function used by all statics
  FG_EXTERN ptrdiff_t BSS_COMPILER_STDCALL fgStaticWndProc(void* hWnd, unsigned int message, size_t wParam, ptrdiff_t lParam);
  // Supplementary fgInitialize function
  FG_EXTERN fgRoot* WinAPIfgInitialize(void* instance);
  // WinAPI destroy function
  FG_EXTERN void WinAPIfgWindow_Destroy(fgWindow* self);
  // WinAPI default behavior function
  //FG_EXTERN char WinAPIfgRoot_Behavior(fgWindow* self, const FG_Msg* msg);
  // UTF8 to UTF16 conversion wrapper. Free t after you finish with it.
  FG_EXTERN void WinAPIutf8to16(wchar_t** t, const char* src);
  // FG_MOVE response function
  FG_EXTERN void WinAPI_FG_MOVE(WinAPIfgWindow* self);
  FG_EXTERN fgStatic* WinAPIfgText_Clone(WinAPIfgText* self);
  FG_EXTERN fgStatic* WinAPIfgImage_Clone(WinAPIfgImage* self);
  
  typedef struct {
    fgRoot root;
    void* instance;
  } WinAPIfgRoot;

  typedef struct {
    unsigned int message;
    size_t wParam;
    ptrdiff_t lParam;
  } WinAPIMessage;

#ifdef  __cplusplus
}
#endif

#endif
