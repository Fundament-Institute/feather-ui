#include "Style.h"

using namespace fgDotNet;
using namespace System;
using namespace System::Drawing;

Style::Style(fgStyle* s) : _owner(false), styles(s->styles) {}
Style::Style() : _owner(true), styles(0) { } // The styles init just zeroes itself.
Style::~Style() { this->!Style(); }
Style::!Style() { if(_owner) { fgStyle s = { styles }; fgStyle_Destroy(&s); } }

StyleMsg^ Style::AddStyleMsg(const FG_Msg* msg)
{
  fgStyle shim = { styles }; // Because fgStyle just wraps a single pointer, we can store the pointer directly and use a shim to avoid allocating any memory
  fgStyleMsg* m = fgStyle_AddStyleMsg(&shim, msg, 0, 0);
  styles = shim.styles;
  return GenNewManagedPtr<StyleMsg, fgStyleMsg>(m);
}
void Style::RemoveStyleMsg(StyleMsg^ msg) {}

FG_UINT Style::GetName(System::String^ name)
{
  TOCHAR(name);
  return fgStyle_GetName((const char*)pstr);
}

Style::operator fgStyleMsg*(Style^ e) { return e->styles; }