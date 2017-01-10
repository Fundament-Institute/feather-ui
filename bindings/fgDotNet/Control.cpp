#include "Control.h"

using namespace fgDotNet;
using namespace System;
using namespace System::Drawing;

Control::Control(fgElement* p) : Element(p) {}
Control::Control(Element^ parent, Element^ next, System::String^ name, fgFlag flags, UnifiedTransform^ transform, unsigned short units) : Element(parent, next, name, flags, transform, units) {}
void Control::SetContextMenu(Element^ menu) { _p->SetContextMenu(menu); }
Element^ Control::GetContextMenu() { return GenNewManagedPtr<Element, fgElement>(_p->GetContextMenu()); }