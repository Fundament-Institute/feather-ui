#include "Skin.h"
#include "Font.h"
#include "Asset.h"
#include "Style.h"

using namespace fgDotNet;
using namespace System;

SkinElement::SkinElement(fgSkinElement* p) : _p(p) {}

SkinTree::SkinTree(fgSkinTree* p) : _p(p) {}
FG_UINT SkinTree::AddChild(System::String^ type, fgFlag flags, UnifiedTransform^ transform, short units, int order) {
  TOCHAR(type);
  return fgSkinTree_AddChild(&reinterpret_cast<fgSkin*>(_p)->tree, (const char*)pstr, flags, &transform->operator fgTransform(), units, order);
}
bool SkinTree::RemoveChild(FG_UINT child) { return fgSkinTree_RemoveChild(&reinterpret_cast<fgSkin*>(_p)->tree, child) != 0; }
SkinLayout^ SkinTree::GetChild(FG_UINT child) { return GenNewManagedPtr<SkinLayout, fgSkinLayout>(fgSkinTree_GetChild(&reinterpret_cast<fgSkin*>(_p)->tree, child)); }
FG_UINT SkinTree::AddStyle(System::String^ name) { TOCHAR(name); return fgSkinTree_AddStyle(&reinterpret_cast<fgSkin*>(_p)->tree, (const char*)pstr); }
bool SkinTree::RemoveStyle(FG_UINT asset) { return fgSkinTree_RemoveStyle(&reinterpret_cast<fgSkin*>(_p)->tree, asset) != 0; }
Style^ SkinTree::GetStyle(FG_UINT asset) { return GenNewManagedPtr<Style, fgStyle>(fgSkinTree_GetStyle(&reinterpret_cast<fgSkin*>(_p)->tree, asset)); }

SkinLayout::SkinLayout(fgSkinLayout* p) : SkinElement(reinterpret_cast<fgSkinElement*>(p)) {}

Skin^ SkinBase::AddSkin(System::String^ name) { TOCHAR(name); return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_AddSkin(_p, (const char*)pstr)); }
bool SkinBase::RemoveSkin(System::String^ name) { TOCHAR(name); return fgSkinBase_RemoveSkin(_p, (const char*)pstr) != 0; }
Skin^ SkinBase::GetSkin(System::String^ name) { TOCHAR(name); return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_GetSkin(_p, (const char*)pstr)); }
Asset^ SkinBase::AddAssetFile(fgFlag flags, System::String^ file) { TOCHAR(file); return GenNewManagedPtr<Asset, void>(fgSkinBase_AddAssetFile(_p, flags, (const char*)pstr)->asset); }
bool SkinBase::RemoveAsset(Asset^ asset) { return fgSkinBase_RemoveAsset(_p, asset) != 0; }
Font^ SkinBase::AddFont(fgFlag flags, System::String^ families, short weight, char italic, unsigned int size) { TOCHAR(families); return GenNewManagedPtr<Font, void>(fgSkinBase_AddFont(_p, flags, (const char*)pstr, weight, italic, size)->font); }
bool SkinBase::RemoveFont(Font^ font) { return fgSkinBase_RemoveFont(_p, font) != 0; }

//Skin^ SkinBase::LoadFileUBJSON(System::String^ file) { TOCHAR(file); return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_LoadFileUBJSON(_p, (const char*)pstr)); }
//Skin^ SkinBase::LoadUBJSON(cli::array<System::Byte>^ data) { pin_ptr<const unsigned char> pin = &data[0]; return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_LoadUBJSON(_p, pin, data->Length)); }
Skin^ SkinBase::LoadFileXML(System::String^ file) { TOCHAR(file); return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_LoadFileXML(_p, (const char*)pstr)); }
Skin^ SkinBase::LoadXML(cli::array<System::Byte>^ data) { pin_ptr<const unsigned char> pin = &data[0]; return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_LoadXML(_p, (const char*)pin, data->Length, "")); }
Skin^ SkinBase::LoadXML(cli::array<System::Byte>^ data, System::String^ path) { TOCHAR(path); pin_ptr<const unsigned char> pin = &data[0]; return GenNewManagedPtr<Skin, fgSkin>(fgSkinBase_LoadXML(_p, (const char*)pin, data->Length, (const char*)pstr)); }

SkinBase::SkinBase(fgSkinBase* p, bool owner) : _p(p), _owner(p) {}

Skin::Skin(fgSkin* p) : SkinBase(&p->base, false) {}
Skin::Skin(System::String^ name) : SkinBase(reinterpret_cast<fgSkinBase*>(new fgSkin()), true) { TOCHAR(name); fgSkin_Init(reinterpret_cast<fgSkin*>(_p), (const char*)pstr); }
Skin::~Skin() { this->!Skin(); }
Skin::!Skin()
{ 
  if(_p && _owner)
  { 
    fgSkin_Destroy(reinterpret_cast<fgSkin*>(_p));
    delete reinterpret_cast<fgSkin*>(_p);
  }
}
Skin^ Skin::GetSkin(System::String^ name) { TOCHAR(name); return GenNewManagedPtr<Skin, fgSkin>(fgSkin_GetSkin(reinterpret_cast<fgSkin*>(_p), (const char*)pstr)); }

Skin::operator fgSkin*(Skin^ e) { return reinterpret_cast<fgSkin*>(e->_p); }
