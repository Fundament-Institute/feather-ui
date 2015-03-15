// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgImplementation.h"

FG_EXTERN void FG_FASTCALL fgLoadImplementation(fgImplementation* fg)
{
  fg->fgLoadImageData = &fgLoadImageData;
  fg->fgLoadText = &fgLoadText;
  fg->fgLoadImageDef = &fgLoadImageDef;
  fg->fgLoadTextDef = &fgLoadTextDef;
  fg->fgEmptyStatic = &fgEmptyStatic;
  fg->fgLoadDefaultText = &fgLoadDefaultText;
  fg->fgLoadDef = &fgLoadDef;
  fg->fgDestroyDef = &fgDestroyDef;
  fg->fgButton_Create = &fgButton_Create;
  fg->fgMenu_Create = &fgMenu_Create;
  fg->fgTopWindow_Create = &fgTopWindow_Create;
  fg->fgInitialize = &fgInitialize;
  fg->fgSingleton = &fgSingleton;
}