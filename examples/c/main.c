// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWindow.h"
#include "fgRoot.h"
#include "fgLayout.h"
#include <stdlib.h>

void statelistener(fgElement* self, const FG_Msg* msg)
{
  fgElement* progbar = fgRoot_GetID(fgSingleton(), "#progbar");
  fgFloatMessage(progbar, FG_SETVALUE, FGVALUE_FLOAT, fgGetFloatMessage(self, FG_GETVALUE, FGVALUE_FLOAT, 0) / fgGetFloatMessage(self, FG_GETRANGE, FGVALUE_FLOAT, 0), 0);
  char buf[10];
  _itoa_s(fgIntMessage(self, FG_GETVALUE, 0, 0), buf, 10, 10);
  fgVoidMessage(progbar, FG_SETTEXT, buf, 0);
}
void makepressed(fgElement* self, const FG_Msg* msg)
{
  fgVoidMessage(self, FG_SETTEXT, "Pressed!", 0);
}

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

  fgRegisterFunction("statelistener", &statelistener);
  fgRegisterFunction("makepressed", &makepressed);

  fgLayout layout;
  fgLayout_Init(&layout);
  fgLayout_LoadFileXML(&layout, "../media/feathertest.xml");
  fgVoidMessage(&fgSingleton()->gui.element, FG_LAYOUTLOAD, &layout, 0);

  fgElement* tabfocus = fgRoot_GetID(fgSingleton(), "#tabfocus");
  if(tabfocus)
  {
    fgElement* tabbutton = (fgElement*)fgVoidMessage(tabfocus, FG_GETSELECTEDITEM, 0, 0);
    FG_Msg m = { FG_ACTION, 0 };
    fgSendMessageAsync(tabbutton, &m, 0, 0);
  }

  while(fgSingleton()->backend.fgProcessMessages());

  fgLayout_Destroy(&layout);
  fgUnloadBackend();
  return 0;
}

struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0, (char**)hInstance);
}