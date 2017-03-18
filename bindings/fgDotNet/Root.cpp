// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgRoot.h"
#include "fgElement.h"
#include "Control.h"
#include "Monitor.h"
#include <vcclr.h>

using namespace System;
using namespace System::Drawing;
using namespace System::Runtime::InteropServices;

namespace fgDotNet {
  delegate void ControlIterator(void* obj, String^ str);
  delegate void ControlListener(fgElement* e, FG_Msg* msg);

  public ref class Root : public Control
  {
  public:
    explicit Root(String^ backend) : Control(0) {
      TOCHAR(backend);
      _p = (fgElement*)fgLoadBackend((const char*)pstr);
      Singleton = this;
    }
    ~Root() {
      if(Singleton == this)
        Singleton = nullptr;
      this->!Root();
    }
    !Root() {
      if(_p)
        ((fgRoot*)_p)->backend.fgTerminate();
      fgUnloadBackend();
    }

    // Returns 0 if handled, 1 otherwise
    size_t Inject(const FG_Msg* msg) { return reinterpret_cast<fgRoot*>(_p)->inject(reinterpret_cast<fgRoot*>(_p), msg); }
    void Update(double delta) { fgRoot_Update(reinterpret_cast<fgRoot*>(_p), delta); }
    void CheckMouseMove() { fgRoot_CheckMouseMove(reinterpret_cast<fgRoot*>(_p)); }
    Monitor^ GetMonitor(RectangleF rect) { return GenNewManagedPtr<Monitor, fgMonitor>(fgRoot_GetMonitor(reinterpret_cast<fgRoot*>(_p), &Element::From(rect))); }
    Element^ GetID(String^ id) { TOCHAR(id); return GenNewManagedPtr<Element, fgElement>(fgRoot_GetID(reinterpret_cast<fgRoot*>(_p), (const char*)pstr)); }
    void AddID(String^ id, Element^ element) { TOCHAR(id); return fgRoot_AddID(reinterpret_cast<fgRoot*>(_p), (const char*)pstr, element); }
    bool RemoveID(Element^ element) { return fgRoot_RemoveID(reinterpret_cast<fgRoot*>(_p), element) != 0; }
    static Element^ Create(String^ type, Element^ parent, Element^ next, String^ name, fgFlag flags, UnifiedTransform^ transform, unsigned short units)
    {
      TOCHAR(type);
      TOCHARSTR(name, pname);
      return GenNewManagedPtr<Element, fgElement>(fgCreate((const char*)pstr, parent, next, (const char*)pname, flags, &transform->operator fgTransform(), units));
    }
    static int RegisterCursor(int cursor, cli::array<Byte>^ data)
    {
      pin_ptr<const unsigned char> pin = &data[0];
      return fgRegisterCursor(cursor, pin, data->Length);
    }
    static int RegisterFunction(String^ name, ControlListener^ del)
    {
      fgListener fn = static_cast<fgListener>(Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(del).ToPointer());
      GC::KeepAlive(del);
      TOCHAR(name);
      return fgRegisterFunction((const char*)pstr, fn);
    }
    //generic<typename T> where T : Element
    //static void RegisterControl(String^ name) { Singleton->controls[name] = typeof(T); }
    //static Type^ GetControl(String^ name) { return safe_cast<Type^>(Singleton->controls[name]); }
    typedef void(*NativeControlIterator)(void*, const char*);
    static void IterateControls(ControlIterator^ del)
    {
      NativeControlIterator fn = static_cast<NativeControlIterator>(Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(del).ToPointer());
      GC::KeepAlive(del);
      fgIterateControls(nullptr, fn);
    }
    bool ProcessMessages()
    {
      return ((fgRoot*)_p)->backend.fgProcessMessages() != 0;
    }
    
    static Root^ Singleton;
  };
}