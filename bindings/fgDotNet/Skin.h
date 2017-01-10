// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgSkin.h"
#include "Element.h"

namespace fgDotNet {
  ref class Skin;
  ref struct StyleLayout;

  public ref class SkinBase
  {
    Skin^ AddSkin(System::String^ name);
    bool RemoveSkin(System::String^ name);
    Skin^ GetSkin(System::String^ name);
    size_t AddAsset(Asset^ asset);
    bool RemoveAsset(FG_UINT asset);
    Asset^ GetAsset(FG_UINT asset);
    size_t AddFont(Font^ font);
    bool RemoveFont(FG_UINT font);
    Font^ GetFont(FG_UINT font);

    Skin^ LoadFileUBJSON(System::String^ file);
    Skin^ LoadUBJSON(cli::array<System::Byte>^ data);
    Skin^ LoadFileXML(System::String^ file);
    Skin^ LoadXML(cli::array<System::Byte>^ data);

  protected:
    SkinBase(fgSkinBase* p, bool owner);
    fgSkinBase* _p;
    bool _owner;
  };

  public ref class Skin : public SkinBase
  {
  public:
    explicit Skin(fgSkin* p);
    Skin();
    ~Skin();
    !Skin();
    FG_UINT AddChild(System::String^ type, System::String^ name, fgFlag flags, UnifiedTransform^ transform, short units, int order);
    bool RemoveChild(FG_UINT child);
    StyleLayout^ GetChild(FG_UINT child);
    FG_UINT AddStyle(System::String^ name);
    bool RemoveStyle(FG_UINT asset);
    Style^ GetStyle(FG_UINT asset);
    Skin^ GetSkin(System::String^ name);

    static operator fgSkin*(Skin^ e);
  };
}