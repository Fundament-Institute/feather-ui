// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgButton.h"
#include "fgTopWindow.h"

fgChild* fgLayoutLoadMapping(const char* name, fgFlag flags, fgChild* parent, fgElement* element)
{
  if(!strcmp(name, "fgChild")) {
    fgChild* r = (fgChild*)malloc(sizeof(fgChild));
    fgChild_Init(r, flags, parent, element);
    return r;
  }
  else if(!strcmp(name, "fgWindow")) {
    fgWindow* r = (fgWindow*)malloc(sizeof(fgWindow));
    fgWindow_Init(r, flags, parent, element);
    return (fgChild*)r;
  }
  else if(!strcmp(name, "fgTopWindow"))
  {
    fgChild* r = fgTopWindow_Create(0, flags, element);
    fgChild_VoidMessage(r, FG_SETPARENT, parent);
    return r;
  }
  else if(!strcmp(name, "fgButton"))
    return fgButton_Create(0, flags, parent, element);

  return 0;
}