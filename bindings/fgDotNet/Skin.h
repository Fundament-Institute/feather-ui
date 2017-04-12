// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgSkin.h"
#include "Style.h"

namespace fgDotNet {
  ref class Skin;

  public ref struct SkinElement
  {
    explicit SkinElement(fgSkinElement* p);
    property System::String^ Type {
      System::String^ get() { return gcnew System::String(_p->type); }
    }
    property UnifiedTransform Transform {
      UnifiedTransform get() { return UnifiedTransform(_p->transform); }
      void set(UnifiedTransform value) { _p->transform = value; }
    }
    property short Units { short get() { return _p->units; } void set(short value) { _p->units = value; } }
    property fgFlag Flags { fgFlag get() { return _p->flags; } void set(fgFlag value) { _p->flags = value; } }
    property Style^ Style { fgDotNet::Style^ get() { return gcnew fgDotNet::Style(&_p->style); } }
    property int Order { int get() { return _p->order; } void set(int value) { _p->order = value; } }
    
  protected:
    fgSkinElement* _p;
  };

  ref struct SkinLayout;

  public ref struct SkinTree
  {
    explicit SkinTree(fgSkinTree* p);
    FG_UINT AddChild(System::String^ type, fgFlag flags, UnifiedTransform^ transform, short units, int order);
    bool RemoveChild(FG_UINT child);
    SkinLayout^ GetChild(FG_UINT child);
    FG_UINT AddStyle(System::String^ name);
    bool RemoveStyle(FG_UINT asset);
    Style^ GetStyle(FG_UINT asset);

  protected:
    fgSkinTree* _p;
  };

  public ref struct SkinLayout : public SkinElement
  {
    explicit SkinLayout(fgSkinLayout* p);
    property SkinTree^ Tree { SkinTree^ get() { return gcnew SkinTree(&reinterpret_cast<fgSkinLayout*>(_p)->tree); } }
    property Element^ Instance { Element^ get() { return GenNewManagedPtr<Element, fgElement>(reinterpret_cast<fgSkinLayout*>(_p)->instance); } }
  };

  public ref class SkinBase
  {
    Skin^ AddSkin(System::String^ name);
    bool RemoveSkin(System::String^ name);
    Skin^ GetSkin(System::String^ name);
    Asset^ AddAssetFile(fgFlag flags, System::String^ file);
    bool RemoveAsset(Asset^ asset);
    Font^ AddFont(fgFlag flags, System::String^ families, short weight, char italic, unsigned int size);
    bool RemoveFont(Font^ font);

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
    explicit Skin(System::String^ name);
    ~Skin();
    !Skin();
    Skin^ GetSkin(System::String^ name);
    property SkinTree^ Tree { SkinTree^ get() { return gcnew SkinTree(&reinterpret_cast<fgSkinLayout*>(_p)->tree); } }

    static operator fgSkin*(Skin^ e);
  };
}