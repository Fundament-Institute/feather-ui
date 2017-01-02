// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

#ifdef BSS_PLATFORM_WIN32
#pragma pack(push)
#pragma pack(8)
#define WINVER 0x0501 //_WIN32_WINNT_WINXP   
#define _WIN32_WINNT 0x0501
#define NTDDI_VERSION 0x05010000 //NTDDI_WINXP
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX // Some compilers enable this by default
#define NOMINMAX
#endif
#include <windows.h>
#pragma pack(pop)
#endif

typedef void(__stdcall *LOADIMPL)(fgImplementation*);

int main(int argc, char** argv)
{
  if(argc < 2)
    return -1;


  HMODULE dll = LoadLibraryA(argv[1]);
  if(!dll)
    return -2;
  
  LOADIMPL loadimpl = (LOADIMPL)GetProcAddress(dll, "fgLoadImplementation");
  if(!loadimpl)
  {
    FreeLibrary(dll);
    return -3;
  }

  fgImplementation impl;
  (*loadimpl)(&impl);
  fgLayoutEditor* editor = LoadLayoutEditor(&impl);
  while(impl.fgMessageLoop(editor->root));
  CloseLayoutEditor(editor);
  FreeLibrary(dll);
}

struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0, (char**)hInstance);
}