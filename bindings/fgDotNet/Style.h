// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgStyle.h"
#include "Element.h"

namespace fgDotNet {
  public value struct StyleMsg
  {
    StyleMsg(fgStyleMsg* s) : msg(&s->msg), next(s->next) {}
    property StyleMsg^ Next {
      StyleMsg^ get() { return GenNewManagedPtr<StyleMsg, fgStyleMsg>(next); }
    }

  private:
    const FG_Msg* msg;
    fgStyleMsg* next;
  };

  public ref struct Style
  {
    Style(fgStyle* s);
    Style();
    ~Style();
    !Style();
    StyleMsg^ AddStyleMsg(const FG_Msg* msg);
    void RemoveStyleMsg(StyleMsg^ msg);

    static FG_UINT GetName(System::String^ name);
    static operator fgStyleMsg*(Style^ e);

  private:
    fgStyleMsg* styles;
    bool _owner;
  };
}