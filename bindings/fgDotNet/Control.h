// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "Element.h"
#include "fgControl.h"

namespace fgDotNet {
  public ref class Control : public Element
  {
  public:
    Control(fgElement* p);
    Control(Element^ parent, Element^ next, System::String^ name, fgFlag flags, UnifiedTransform^ transform, unsigned short units);
    void SetContextMenu(Element^ menu);
    Element^ GetContextMenu();
  };
}