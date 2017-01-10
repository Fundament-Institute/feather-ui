// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgLayout.h"
#include "Skin.h"

namespace fgDotNet {
  public ref class ClassLayout
  {
  public:
    explicit ClassLayout(fgClassLayout* p);
    void AddUserString(System::String^ key, System::String^ value);
    FG_UINT AddChild(System::String^ type, System::String^ name, fgFlag flags, UnifiedTransform^ transform, short units, int order);
    bool RemoveChild(FG_UINT child);
    ClassLayout^ GetChild(FG_UINT child);

  protected:
    fgClassLayout* _p;
  };

  public ref class Layout : public SkinBase
  {
  public:
    Layout(fgLayout* p);
    Layout();
    ~Layout();
    !Layout();
    FG_UINT AddLayout(System::String^ type, System::String^ name, fgFlag flags, UnifiedTransform^ transform, short units, int order);
    bool RemoveLayout(FG_UINT layout);
    ClassLayout^ GetLayout(FG_UINT layout);

    void LoadFileUBJSON(System::String^ file);
    void LoadUBJSON(cli::array<System::Byte>^ data);
    void SaveFileUBJSON(System::String^ file);
    void LoadFileXML(System::String^ file);
    bool LoadXML(cli::array<System::Byte>^ data);
    void SaveFileXML(System::String^ file);

    static void SaveElementXML(Element^ e, System::String^ file);
    static operator fgLayout*(Layout^ e);
  };
}