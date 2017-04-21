#include "Element.h"
#include "Font.h"
#include "Asset.h"
#include "Style.h"
#include "Layout.h"
#include "Skin.h"
#include "Util.h"
#include "fgRoot.h"

using namespace fgDotNet;
using namespace System;
using namespace System::Drawing;

Element::Element(fgElement* p) : _p(p), _owner(false) {}
Element::Element(Element^ parent, Element^ next, System::String^ name, fgFlag flags, UnifiedTransform^ transform, unsigned short units) : _p(nullptr), _owner(true)
{
  TOCHAR(name);
  fgTransform t = (fgTransform&)transform;
  _p = fgCreate("Element", parent, next, (const char*)pstr, flags, &t, units);
}
Element::~Element() { this->!Element(); }
Element::!Element() { if(_owner) VirtualFreeChild(_p); }
void Element::Move(unsigned short subtype, Element^ child, size_t diff) { _p->Move(subtype, child, diff); }
size_t Element::SetAlpha(float alpha) { return _p->SetAlpha(alpha); }
size_t Element::SetArea(UnifiedRect^ area) { return _p->SetArea((CRect&)area); }
size_t Element::SetTransform(UnifiedTransform^ transform) { return _p->SetTransform((fgTransform&)transform); }
void Element::SetFlag(fgFlag flag, bool value) { _p->SetFlag(flag, value); }
void Element::SetFlags(fgFlag flags) { _p->SetFlags(flags); }
size_t Element::SetMargin(RectangleF^ margin) { return _p->SetMargin(From(margin)); }
size_t Element::SetPadding(RectangleF^ padding) { return _p->SetMargin(From(padding)); }
void Element::SetParent(Element^ parent) { _p->SetParent(parent, nullptr); }
void Element::SetParent(Element^ parent, Element^ next) { _p->SetParent(parent, next); }
size_t Element::AddChild(Element^ child) { return _p->AddChild(child, nullptr); }
size_t Element::AddChild(Element^ child, Element^ next) { return _p->AddChild(child, next); }
//Element^ Element::AddItem(void* item, size_t index = (size_t)~0);
Element^ Element::AddItemText(String^ item) { pin_ptr<const wchar_t> p = &item->ToCharArray()[0]; return GenNewManagedPtr<Element, fgElement>(_p->AddItemText((const char*)p, FGTEXTFMT_UTF16)); }
Element^ Element::AddItemElement(Element^ item) { return GenNewManagedPtr<Element, fgElement>(_p->AddItemElement(item == nullptr ? 0 : item->_p)); }
size_t Element::RemoveChild(Element^ child) { return _p->RemoveChild(child); }
size_t Element::RemoveItem(size_t item) { return _p->RemoveItem(item); }
void Element::LayoutChange(unsigned short subtype, Element^ target, Element^ old) { _p->LayoutChange(subtype, target, old); }
size_t Element::LayoutFunction(const FG_Msg& msg, UnifiedRect^ area, bool scrollbar) { return _p->LayoutFunction(msg, (CRect&)area, scrollbar); }
Element^ Element::LayoutLoad(Layout^ layout) { return GenNewManagedPtr<Element, fgElement>(_p->LayoutLoad(layout)); }
size_t Element::DragOver(int x, int y) { return _p->DragOver(x, y); }
size_t Element::Drop(int x, int y, unsigned char allbtn) { return _p->Drop(x, y, allbtn); }
void Element::Draw(RectangleF^ area, DrawAuxData^ aux) { _p->Draw(&From(area), &aux->operator fgDrawAuxData()); }
size_t Element::Inject(const FG_Msg* msg, RectangleF^ area) { return _p->Inject(msg, &From(area)); }
size_t Element::SetSkin(Skin^ skin) { return _p->SetSkin(skin); }
Skin^ Element::GetSkin() { return nullptr; }
Skin^ Element::GetSkin(Element^ child) { return nullptr; }
size_t Element::SetStyle(String^ name, FG_UINT mask) { TOCHAR(name); return _p->SetStyle((const char*)pstr, mask); }
size_t Element::SetStyle(Style^ style) { fgStyle s = fgStyle{ style }; return _p->SetStyle(&s); }
size_t Element::SetStyle(FG_UINT index, FG_UINT mask) { return _p->SetStyle(index, mask); }
Style^ Element::GetStyle() { return GenNewManagedPtr<Style, fgStyle>(_p->GetStyle()); }
Point^ Element::GetDPI() { fgIntVec& p = _p->GetDPI(); return Point(p.x, p.y); }
void Element::SetDPI(int x, int y) { _p->SetDPI(x, y); }
String^ Element::GetClassName() { return gcnew System::String(_p->GetClassName()); }
size_t Element::GetUserdata() { return reinterpret_cast<size_t>(_p->GetUserdata(0)); }
size_t Element::GetUserdata(String^ name) { TOCHAR(name); return reinterpret_cast<size_t>(_p->GetUserdata((const char*)pstr)); }
void Element::SetUserdata(size_t data) { _p->SetUserdata(reinterpret_cast<void*>(data)); }
void Element::SetUserdata(size_t data, String^ name) { TOCHAR(name); _p->SetUserdata(reinterpret_cast<void*>(data), (const char*)pstr); }
size_t Element::MouseDown(int x, int y, unsigned char button, unsigned char allbtn) { return _p->MouseDown(x, y, button, allbtn); }
size_t Element::MouseDblClick(int x, int y, unsigned char button, unsigned char allbtn) { return _p->MouseDblClick(x, y, button, allbtn); }
size_t Element::MouseUp(int x, int y, unsigned char button, unsigned char allbtn) { return _p->MouseUp(x, y, button, allbtn); }
size_t Element::MouseOn(int x, int y) { return _p->MouseOn(x,y); }
size_t Element::MouseOff(int x, int y) { return _p->MouseOff(x, y); }
size_t Element::MouseMove(int x, int y) { return _p->MouseMove(x, y); }
size_t Element::MouseScroll(int x, int y, unsigned short delta, unsigned short hdelta) { return _p->MouseScroll(x, y, delta, hdelta); }
size_t Element::KeyUp(unsigned char keycode, char sigkeys) { return _p->KeyUp(keycode, sigkeys); }
size_t Element::KeyDown(unsigned char keycode, char sigkeys) { return _p->KeyDown(keycode, sigkeys); }
size_t Element::KeyChar(int keychar, char sigkeys) { return _p->KeyChar(keychar, sigkeys); }
size_t Element::JoyButtonDown(short joybutton) { return _p->JoyButtonDown(joybutton); }
size_t Element::JoyButtonUp(short joybutton) { return _p->JoyButtonUp(joybutton); }
size_t Element::JoyAxis(float joyvalue, short joyaxis) { return _p->JoyAxis(joyvalue, joyaxis); }
size_t Element::GotFocus() { return _p->GotFocus(); }
void Element::LostFocus() { _p->LostFocus(); }
size_t Element::SetName(String^ name) { TOCHAR(name); return _p->SetName((const char*)pstr); }
String^ Element::GetName() { return gcnew String(_p->GetName()); }

void Element::SetContextMenu(Element^ menu) { _p->SetContextMenu(menu == nullptr ? 0 : menu->_p); }
Element^ Element::GetContextMenu() { return GenNewManagedPtr<Element, fgElement>(_p->GetContextMenu()); }
void Element::Neutral() { _p->Neutral(); }
void Element::Hover() { _p->Hover(); }
void Element::Active() { _p->Active(); }
void Element::Action() { _p->Action(); }
void Element::SetMaxDim(float x, float y) { _p->SetDim(x, y, FGDIM_MAX); }
void Element::SetMinDim(float x, float y) { _p->SetDim(x, y, FGDIM_MIN); }
PointF^ Element::GetMaxDim() { const AbsVec* dim = _p->GetDim(FGDIM_MAX); return !dim ? nullptr : gcnew PointF(dim->x, dim->y); }
PointF^ Element::GetMinDim() { const AbsVec* dim = _p->GetDim(FGDIM_MIN); return !dim ? nullptr : gcnew PointF(dim->x, dim->y); }
Element^ Element::GetItem(size_t index) { return GenNewManagedPtr<Element, fgElement>(_p->GetItem(index)); }
Element^ Element::GetItemAt(int x, int y) { return GenNewManagedPtr<Element, fgElement>(_p->GetItemAt(x,y)); }
size_t Element::GetNumItems() { return _p->GetNumItems(); }
Element^ Element::GetSelectedItem() { return gcnew Element(_p->GetSelectedItem(0)); }
Element^ Element::GetSelectedItem(size_t index) { return GenNewManagedPtr<Element, fgElement>(_p->GetSelectedItem(index)); }
size_t Element::GetValue(ptrdiff_t aux) { return _p->GetValue(aux); }
float Element::GetValueF(ptrdiff_t aux) { return _p->GetValueF(aux); }
size_t Element::SetValue(ptrdiff_t value) { return _p->SetValue(value); }
size_t Element::SetValueF(float value) { return _p->SetValueF(value); }
size_t Element::SetAsset(Asset^ asset) { return _p->SetAsset(asset); }
size_t Element::SetUV(UnifiedRect^ uv) { return _p->SetUV(uv->operator CRect()); }
size_t Element::SetColor(unsigned int color, FGSETCOLOR index) { return _p->SetColor(color, index); }
size_t Element::SetOutline(float outline) { return _p->SetOutline(outline); }
size_t Element::SetFont(fgDotNet::Font^ font) { return _p->SetFont(font); }
size_t Element::SetLineHeight(float lineheight) { return _p->SetLineHeight(lineheight); }
size_t Element::SetLetterSpacing(float letterspacing) { return _p->SetLetterSpacing(letterspacing); }
size_t Element::SetText(String^ text) { pin_ptr<const wchar_t> p = &text->ToCharArray()[0]; return _p->SetTextW(p); }
size_t Element::SetPlaceholder(String^ text) { pin_ptr<const wchar_t> p = &text->ToCharArray()[0]; return _p->SetPlaceholderW(p); }
size_t Element::SetMask(wchar_t mask) { return _p->SetMask(mask); }
Asset^ Element::GetAsset() { return GenNewManagedPtr<Asset, void>(_p->GetAsset()); }
UnifiedRect^ Element::GetUV() { const CRect* r = _p->GetUV(); return !r ? nullptr : gcnew UnifiedRect(*r); }
unsigned int Element::GetColor(FGSETCOLOR index) { return _p->GetColor(index); }
float Element::GetOutline() { return _p->GetOutline(); }
fgDotNet::Font^ Element::GetFont() { return GenNewManagedPtr<Font, void>(_p->GetFont()); }
float Element::GetLineHeight() { return _p->GetLineHeight(); }
float Element::GetLetterSpacing() { return _p->GetLetterSpacing(); }
String^ Element::GetText() { return gcnew System::String(_p->GetTextW()); }
String^ Element::GetPlaceholder() { return gcnew System::String(_p->GetPlaceholderW()); }
wchar_t Element::GetMask() { return (wchar_t)_p->GetMask(); }
void Element::AddListener(unsigned short type, fgListener listener) { _p->AddListener(type, listener); }
Element^ Element::GetChildByName(System::String^ name) { return GenNewManagedPtr<Element, fgElement>(_p->GetChildByName(StringToUTF8(name).c_str())); }

Element::operator fgElement*(Element^ e) { return e->_p; }
AbsRect Element::From(System::Drawing::RectangleF^ r) { return AbsRect{ r->Left, r->Top, r->Right, r->Bottom }; }
AbsVec Element::From(System::Drawing::PointF^ p) { return AbsVec{ p->X, p->Y }; }
fgIntVec Element::From(System::Drawing::Point^ p) { return fgIntVec{ p->X, p->Y }; }