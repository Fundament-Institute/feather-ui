// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

int main(int argc, char** argv)
{
#if defined(BSS_DEBUG) && defined(BSS_CPU_x86_64)
  fgLoadBackend("fgDirect2D_d.dll");
#elif defined(BSS_CPU_x86_64)
  fgLoadBackend("fgDirect2D.dll");
#elif defined(BSS_DEBUG)
  fgLoadBackend("fgDirect2D32_d.dll");
#else
  fgLoadBackend("fgDirect2D32.dll");
#endif

  fgLayoutEditor editor;
  fgLayoutEditor_Init(&editor);
  while(fgSingleton()->backend.fgProcessMessages());
  fgLayoutEditor_Destroy(&editor);

  fgUnloadBackend();
}

struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0, (char**)hInstance);
}