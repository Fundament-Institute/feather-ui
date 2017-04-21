#include "Layout.h"

using namespace fgDotNet;
using namespace System;
using namespace System::Drawing;

ClassLayout::ClassLayout(fgClassLayout* p) : _p(p) {}
void ClassLayout::AddUserString(System::String^ key, System::String^ value) { TOCHAR(key); TOCHARSTR(value, pvalue); _p->AddUserString((const char*)pstr, (const char*)pvalue); }
FG_UINT ClassLayout::AddChild(System::String^ type, System::String^ name, fgFlag flags, UnifiedTransform^ transform, short units, int order)
{
  TOCHAR(type);
  TOCHARSTR(name, pname);
  return fgClassLayout_AddChild(_p, (const char*)pstr, (const char*)pname, flags, &transform->operator fgTransform(), units, order);
}
bool ClassLayout::RemoveChild(FG_UINT child) { return _p->RemoveChild(child) != 0; }
ClassLayout^ ClassLayout::GetChild(FG_UINT child) { return GenNewManagedPtr<ClassLayout, fgClassLayout>(_p->GetChild(child)); }

Layout::Layout(fgLayout* p) : SkinBase(&p->base, false) {}
Layout::Layout() : SkinBase(reinterpret_cast<fgSkinBase*>(new fgLayout()), true) { fgLayout_Init(reinterpret_cast<fgLayout*>(_p)); }
Layout::~Layout() { this->!Layout(); }
Layout::!Layout() { if(_p && _owner) { fgLayout_Destroy(reinterpret_cast<fgLayout*>(_p)); delete reinterpret_cast<fgLayout*>(_p); } }
FG_UINT Layout::AddChild(System::String^ type, System::String^ name, fgFlag flags, UnifiedTransform^ transform, short units, int order)
{
  TOCHAR(type);
  TOCHARSTR(name, pname);
  return fgLayout_AddChild(reinterpret_cast<fgLayout*>(_p), (const char*)pstr, (const char*)pname, flags, &transform->operator fgTransform(), units, order);
}
bool Layout::RemoveChild(FG_UINT layout) { return reinterpret_cast<fgLayout*>(_p)->RemoveChild(layout) != 0; }
ClassLayout^ Layout::GetChild(FG_UINT layout) { return GenNewManagedPtr<ClassLayout, fgClassLayout>(reinterpret_cast<fgLayout*>(_p)->GetChild(layout)); }
Layout^ Layout::AddLayout(System::String^ name) { TOCHAR(name); return GenNewManagedPtr<Layout, fgLayout>(reinterpret_cast<fgLayout*>(_p)->AddLayout((const char*)pstr)); }
bool Layout::RemoveLayout(System::String^ name) { TOCHAR(name); return reinterpret_cast<fgLayout*>(_p)->RemoveLayout((const char*)pstr); }
Layout^ Layout::GetLayout(System::String^ name) { TOCHAR(name); return GenNewManagedPtr<Layout, fgLayout>(reinterpret_cast<fgLayout*>(_p)->GetLayout((const char*)pstr)); }

void Layout::LoadFileUBJSON(System::String^ file) { TOCHAR(file); reinterpret_cast<fgLayout*>(_p)->LoadFileUBJSON((const char*)pstr); }
void Layout::LoadUBJSON(cli::array<System::Byte>^ data) { pin_ptr<const unsigned char> p = &data[0]; return reinterpret_cast<fgLayout*>(_p)->LoadUBJSON((const char*)p, data->Length); }
void Layout::SaveFileUBJSON(System::String^ file) { TOCHAR(file); reinterpret_cast<fgLayout*>(_p)->SaveFileUBJSON((const char*)pstr); }
void Layout::LoadFileXML(System::String^ file) { TOCHAR(file); reinterpret_cast<fgLayout*>(_p)->LoadFileXML((const char*)pstr); }
bool Layout::LoadXML(cli::array<System::Byte>^ data) { pin_ptr<const unsigned char> p = &data[0]; return reinterpret_cast<fgLayout*>(_p)->LoadXML((const char*)p, data->Length) != 0; }
void Layout::SaveFileXML(System::String^ file) { TOCHAR(file); reinterpret_cast<fgLayout*>(_p)->SaveFileXML((const char*)pstr); }

Layout::operator fgLayout*(Layout^ e) { return reinterpret_cast<fgLayout*>(e->_p); }
