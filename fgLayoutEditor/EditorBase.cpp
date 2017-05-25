// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "EditorBase.h"
#include "bss-util/bss_util.h"
#include "bss-util/Trie.h"

using namespace bss;

EditorBase::EditorBase(fgLayout* layout)
{
  bssFill(*this, 0);
  fgLayout_Init(&curlayout, 0);
}
EditorBase::~EditorBase()
{
  if(_window)
    VirtualFreeChild(_window);
  fgLayout_Destroy(&curlayout);
}
void EditorBase::WindowOnDestroy(struct _FG_ELEMENT* e, const FG_Msg*)
{
  _window = 0; // Don't delete the window twice
  Destroy();
}

uint32_t EditorBase::ParseColor(const char* s)
{
  fgColor color;
  char* r;
  color.color = strtoul(s, &r, 16);
  if(strlen(s) == 8 && s[0] == '0' && s[1] == 'x') // If this is true, it's a non-alpha color value
  {
    std::swap(color.r, color.b);
    color.a = 255;
  }
  else
  {
    std::swap(color.r, color.a);
    std::swap(color.g, color.b);
  }
  return color.color;
}

uint16_t EditorBase::GetTransformMsg(const fgStyle& target, fgTransform& out)
{
  for(fgStyleMsg* m = target.styles; m != 0; m = m->next)
  {
    if(m->msg.type == FG_SETAREA)
      out.area = *(CRect*)m->msg.p;
    else if(m->msg.type == FG_SETTRANSFORM)
      out = *(fgTransform*)m->msg.p;
    else
      continue;
    return m->msg.subtype;
  }
  return 0;
}

void EditorBase::ParseStyleMsg(fgStyle& target, fgElement* instance, fgSkinElement* element, fgClassLayout* layout, PROPERTIES id, const char* s)
{
  FG_Msg m = { FG_CUSTOMEVENT };
  char* out;
  if(s && !s[0])
    s = 0;

  switch(id)
  {
  case PROP_USERID:
    if(layout)
      layout->userid = strtol(s, &out, 10);
    break;
  case PROP_TEXT:
    RemoveStyleMsg(target, FG_SETTEXT, FGTEXTFMT_UTF8);
    RemoveStyleMsg(target, FG_SETTEXT, FGTEXTFMT_UTF16);
    RemoveStyleMsg(target, FG_SETTEXT, FGTEXTFMT_UTF32);
    if(!s)
      return;
    m.type = FG_SETTEXT;
    m.subtype = FGTEXTFMT_UTF8;
    m.p = const_cast<char*>(s);
    target.AddStyleMsg(&m, strlen(s) + 1);
    break;
  case PROP_PLACEHOLDER:
    RemoveStyleMsg(target, FG_SETTEXT, FGTEXTFMT_PLACEHOLDER_UTF8);
    RemoveStyleMsg(target, FG_SETTEXT, FGTEXTFMT_PLACEHOLDER_UTF16);
    RemoveStyleMsg(target, FG_SETTEXT, FGTEXTFMT_PLACEHOLDER_UTF32);
    if(!s)
      return;
    m.type = FG_SETTEXT;
    m.subtype = FGTEXTFMT_PLACEHOLDER_UTF8;
    m.p = const_cast<char*>(s);
    target.AddStyleMsg(&m, strlen(s) + 1);
    break;
  case PROP_FONT:
    RemoveStyleMsg(target, FG_SETFONT);
    break;
  case PROP_LINEHEIGHT:
    RemoveStyleMsg(target, FG_SETLINEHEIGHT);
    if(!s)
      return;
    m.type = FG_SETLINEHEIGHT;
    m.f = strtof(s, &out);
    target.AddStyleMsg(&m);
    break;
  case PROP_LETTERSPACING:
    RemoveStyleMsg(target, FG_SETLETTERSPACING);
    if(!s)
      return;
    m.type = FG_SETLETTERSPACING;
    m.f = strtof(s, &out);
    target.AddStyleMsg(&m);
    break;
  case PROP_COLOR:
  case PROP_PLACECOLOR:
  case PROP_CURSORCOLOR:
  case PROP_SELECTCOLOR:
  case PROP_HOVERCOLOR:
  case PROP_DRAGCOLOR:
  case PROP_EDGECOLOR:
  case PROP_DIVIDERCOLOR:
  case PROP_COLUMNDIVIDERCOLOR:
  case PROP_ROWEVENCOLOR:
    RemoveStyleMsg(target, FG_SETCOLOR, m.subtype);
    if(!s)
      return;
    m.type = FG_SETCOLOR;
    m.subtype = (id - PROP_COLOR);
    m.u = ParseColor(s);
    target.AddStyleMsg(&m);
    break;
  case PROP_VALUEI:
    RemoveStyleMsg(target, FG_SETVALUE);
    if(!s)
      return;
    m.type = FG_SETVALUE;
    m.subtype = FGVALUE_INT64;
    m.i = strtol(s, &out, 10);
    target.AddStyleMsg(&m);
    break;
  case PROP_VALUEF:
    RemoveStyleMsg(target, FG_SETVALUE);
    if(!s)
      return;
    m.type = FG_SETVALUE;
    m.subtype = FGVALUE_FLOAT;
    m.f = strtof(s, &out);
    target.AddStyleMsg(&m);
    break;
  case PROP_RANGE:
    RemoveStyleMsg(target, FG_SETRANGE);
    if(!s)
      return;
    m.type = FG_SETRANGE;
    m.subtype = FGVALUE_FLOAT;
    m.f = strtof(s, &out);
    target.AddStyleMsg(&m);
    break;
  case PROP_UV:
    RemoveStyleMsg(target, FG_SETUV);
    if(!s)
      return;
    m.type = FG_SETUV;
    m.p = alloca(sizeof(CRect));
    m.subtype = fgStyle_ParseCRect(s, (CRect*)m.p);
    target.AddStyleMsg(&m, sizeof(CRect));
    break;
  case PROP_ASSET:
    RemoveStyleMsg(target, FG_SETASSET);
    if(!s)
      return;
    break;
  case PROP_OUTLINE:
    RemoveStyleMsg(target, FG_SETOUTLINE);
    if(!s)
      return;
    m.type = FG_SETOUTLINE;
    m.f = strtof(s, &out);
    target.AddStyleMsg(&m);
    break;
  case PROP_CONTEXTMENU:
    RemoveStyleMsg(target, FG_SETCONTEXTMENU);
    if(!s)
      return;
    break;
  case PROP_MARGIN:
    RemoveStyleMsg(target, FG_SETMARGIN);
    if(!s)
      return;
    m.p = ALLOCA(sizeof(AbsRect));
    m.type = FG_SETMARGIN;
    m.subtype = fgStyle_ParseAbsRect(s, (AbsRect*)m.p);
    target.AddStyleMsg(&m, sizeof(AbsRect));
    break;
  case PROP_PADDING:
    RemoveStyleMsg(target, FG_SETPADDING);
    if(!s)
      return;
    m.p = ALLOCA(sizeof(AbsRect));
    m.type = FG_SETPADDING;
    m.subtype = fgStyle_ParseAbsRect(s, (AbsRect*)m.p);
    target.AddStyleMsg(&m, sizeof(AbsRect));
    break;
  case PROP_MINDIM:
  case PROP_MAXDIM:
    m.type = FG_SETDIM;
    m.subtype = (id == PROP_MINDIM) ? FGDIM_MIN : FGDIM_MAX;
    for(fgStyleMsg* msg = target.styles; msg != 0; msg = msg->next)
    {
      if(msg->msg.type == FG_SETDIM && (msg->msg.subtype & 3) == m.subtype)
        target.RemoveStyleMsg(msg);
    }
    if(!s)
      return;
    {
      AbsVec dim;
      fgStyle_ParseAbsVec(s, &dim);
      m.f = dim.x;
      m.f2 = dim.y;
    }
    target.AddStyleMsg(&m);
    break;
  case PROP_SCALING:
    RemoveStyleMsg(target, FG_SETSCALING);
    if(!s)
      return;
    m.type = FG_SETSCALING;
    {
      AbsVec scale;
      fgStyle_ParseAbsVec(s, &scale);
      m.f = scale.x;
      m.f2 = scale.y;
    }
    target.AddStyleMsg(&m);
    break;
  case PROP_SKIN:
    break;
  case PROP_ID:
    if(layout && s)
    {
      if(layout->id)
        fgFreeText(layout->id, __FILE__, __LINE__);
      layout->id = fgCopyText(s, __FILE__, __LINE__);
    }
    break;
  case PROP_NAME:
    if(layout && s)
    {
      if(layout->name)
        fgFreeText(layout->name, __FILE__, __LINE__);
      layout->name = fgCopyText(s, __FILE__, __LINE__);
    }
    break;
  case PROP_AREA:
  case PROP_ROTATION:
  case PROP_CENTER:
    m.type = FG_SETTRANSFORM;
    if(!element)
    {
      m.p = ALLOCA(sizeof(fgTransform));
      *(fgTransform*)m.p = fgTransform_EMPTY;
      m.subtype = GetTransformMsg(target, *(fgTransform*)m.p);
    }
    RemoveStyleMsg(target, FG_SETAREA);
    RemoveStyleMsg(target, FG_SETTRANSFORM);
    if(!s)
      return;
    if(element)
    {
      m.p = &element->transform;
      m.subtype = element->units;
    }

    switch(id)
    {
    case PROP_AREA:
      m.subtype = fgStyle_ParseCRect(s, &((fgTransform*)m.p)->area) | (m.subtype&(FGUNIT_X_MASK | FGUNIT_Y_MASK));
      break;
    case PROP_ROTATION:
      ((fgTransform*)m.p)->rotation = strtof(s, &out);
      break;
    case PROP_CENTER:
      m.subtype = fgStyle_ParseCVec(s, &((fgTransform*)m.p)->center) | (m.subtype&(FGUNIT_LEFT_MASK | FGUNIT_TOP_MASK | FGUNIT_RIGHT_MASK | FGUNIT_BOTTOM_MASK));
      break;
    }

    if(element)
      element->units = m.subtype;
    else if(s) // Only add the message to the style if we didn't delete it entirely
      target.AddStyleMsg(&m, sizeof(fgTransform));
    break;
  case PROP_STYLE:
    RemoveStyleMsg(target, FG_SETSTYLE);
    if(!s)
      return;
    m.type = FG_SETSTYLE;
    m.subtype = FGSETSTYLE_INDEX;
    m.i = strtol(s, &out, 10);
    target.AddStyleMsg(&m);
    break;
  case PROP_ORDER:
    if(element && s)
      element->order = strtol(s, &out, 10);
    break;
  case PROP_FLAGS:
    //RemoveStyleMsg(target, FG_SETFLAGS);
    //RemoveStyleMsg(target, FG_SETFLAG);
    if(!s)
      return;
    //if(element)
    //{
    //  element->flags = 
    //}
    break;
  case PROP_ALPHA:
    RemoveStyleMsg(target, FG_SETALPHA);
    if(!s)
      return;
    m.type = FG_SETALPHA;
    m.f = strtof(s, &out);
    target.AddStyleMsg(&m);
    break;
  }

  if(m.type != FG_CUSTOMEVENT && instance != 0)
    fgSendMessage(instance, &m);
}
void EditorBase::RemoveStyleMsg(fgStyle& s, uint16_t type, uint16_t subtype)
{
  for(fgStyleMsg* m = s.styles; m != 0;)
  {
    fgStyleMsg* hold = m;
    m = m->next;
    if(hold->msg.type == type && (subtype == (uint16_t)~0 || hold->msg.subtype == subtype))
      s.RemoveStyleMsg(hold);
  }
}

fgElement* EditorBase::AddProp(fgGrid& g, const char* name, const char* type, FG_UINT userid, fgMessage fn, fgFlag flags)
{
  static const fgTransform TF1 = { { 0,0,0,0,0,0.5,20,0 }, 0,{ 0,0,0,0 } };
  static const fgTransform TF2 = { { 0,0.5,0,0,0,1,20,0 }, 0,{ 0,0,0,0 } };
  fgGridRow* row = g.InsertRow();
  fgCreate("Text", *row, 0, 0, FGELEMENT_EXPANDY, &fgTransform_EMPTY, 0)->SetText(name);
  fgElement* prop = fgCreate(type, *row, 0, 0, flags, &fgTransform_EMPTY, 0);
  prop->userid = userid;
  if(fn)
    prop->message = fn;
  return prop;
}

void EditorBase::AddMutableProp(fgGrid& g, PROPERTIES id, const char* type, std::function<void(fgElement*, const char*)>& f, fgFlag flags)
{
  const char* names[PROP_TOTALPLUSONE - 1] = { "UserID", "UserInfo", "Text", "Placeholder", "Font", "Line Height", "Letter Spacing", "Color",
    "Color:Placeholder", "Color:Cursor", "Color:Select", "Color:Hover", "Color:Drag","Color:Edge", "Color:Divider", "Color:ColumnDivider", "Value",
    "Value", "Range", "UV", "Asset", "Outline", "Splitter", "Context Menu" };

  f(AddProp(g, names[id - 1], type, id, 0, flags), type);
}

void EditorBase::ClearProps(fgGrid& g)
{
  for(size_t i = 0; i < 17; ++i)
    g.GetRow(i)->GetItem(1)->SetText("");

  while(g.GetNumRows() > 17)
    g.RemoveRow(g.GetNumRows() - 1);
}

void EditorBase::LoadProps(fgGrid& g, const char* type, fgClassLayout* layout, fgSkinElement* element, fgStyle& style, std::function<void(fgElement*, const char*)>& f)
{
  static Trie<uint16_t, true> t(31, "element", "control", "scrollbar", "box", "list", "grid", "resource", "text", "button", "window", "checkbox", "radiobutton",
    "progressbar", "slider", "textbox", "treeview", "treeitem", "listitem", "curve", "dropdown", "tabcontrol", "menu", "submenu",
    "menuitem", "gridrow", "workspace", "toolbar", "toolgroup", "combobox", "debug", "skin");

  ClearProps(g);
  if(type)
  {
    switch(t[type])
    {
    case 0: // element
    default:
      break;
    case 11: // radiobutton
    case 10: // checkbox
      AddMutableProp(g, PROP_VALUEI, "textbox", f);
    case 7: // text
    case 8: // button
    case 9: // window
    case 28: // combobox
      AddMutableProp(g, PROP_TEXT, "textbox", f);
      AddMutableProp(g, PROP_COLOR, "textbox", f);
      AddMutableProp(g, PROP_FONT, "textbox", f);
      AddMutableProp(g, PROP_LINEHEIGHT, "textbox", f);
      AddMutableProp(g, PROP_LETTERSPACING, "textbox", f);
      break;
    case 12: // progressbar
      AddMutableProp(g, PROP_VALUEF, "textbox", f);
      AddMutableProp(g, PROP_TEXT, "textbox", f);
      AddMutableProp(g, PROP_COLOR, "textbox", f);
      AddMutableProp(g, PROP_FONT, "textbox", f);
      AddMutableProp(g, PROP_LINEHEIGHT, "textbox", f);
      AddMutableProp(g, PROP_LETTERSPACING, "textbox", f);
      break;
    case 14: // textbox
      AddMutableProp(g, PROP_TEXT, "textbox", f, FGELEMENT_EXPANDY | FGTEXTBOX_ACTION);
      AddMutableProp(g, PROP_PLACEHOLDER, "textbox", f);
      AddMutableProp(g, PROP_COLOR, "textbox", f);
      AddMutableProp(g, PROP_PLACECOLOR, "textbox", f);
      AddMutableProp(g, PROP_CURSORCOLOR, "textbox", f);
      AddMutableProp(g, PROP_SELECTCOLOR, "textbox", f);
      AddMutableProp(g, PROP_FONT, "textbox", f);
      AddMutableProp(g, PROP_LINEHEIGHT, "textbox", f);
      AddMutableProp(g, PROP_LETTERSPACING, "textbox", f);
      break;
    case 19: // dropdown
      AddMutableProp(g, PROP_SELECTCOLOR, "textbox", f);
      AddMutableProp(g, PROP_HOVERCOLOR, "textbox", f);
      break;
    case 4: // list
      AddMutableProp(g, PROP_DRAGCOLOR, "textbox", f);
      AddMutableProp(g, PROP_VALUEF, "textbox", f);
      break;
    case 5: // grid
      AddMutableProp(g, PROP_DRAGCOLOR, "textbox", f);
      AddMutableProp(g, PROP_VALUEF, "textbox", f);
      AddMutableProp(g, PROP_RANGE, "textbox", f);
      break;
    case 23: // menuitem:
      AddMutableProp(g, PROP_TEXT, "textbox", f);
      break;
    case 6: // resource
      AddMutableProp(g, PROP_UV, "textbox", f);
      AddMutableProp(g, PROP_ASSET, "textbox", f);
      AddMutableProp(g, PROP_OUTLINE, "textbox", f);
      AddMutableProp(g, PROP_COLOR, "textbox", f);
      AddMutableProp(g, PROP_EDGECOLOR, "textbox", f);
      break;
    case 18: // curve
      AddMutableProp(g, PROP_COLOR, "textbox", f);
      AddMutableProp(g, PROP_VALUEF, "textbox", f);
      break;
    case 13: // slider
      AddMutableProp(g, PROP_VALUEF, "textbox", f);
      AddMutableProp(g, PROP_RANGE, "textbox", f);
      break;
    case 30: // skin
      AddMutableProp(g, PROP_TEXT, "textbox", f, FGELEMENT_EXPANDY | FGTEXTBOX_ACTION);
      AddMutableProp(g, PROP_PLACEHOLDER, "textbox", f);
      AddMutableProp(g, PROP_COLOR, "textbox", f);
      AddMutableProp(g, PROP_PLACECOLOR, "textbox", f);
      AddMutableProp(g, PROP_CURSORCOLOR, "textbox", f);
      AddMutableProp(g, PROP_SELECTCOLOR, "textbox", f);
      AddMutableProp(g, PROP_HOVERCOLOR, "textbox", f);
      AddMutableProp(g, PROP_EDGECOLOR, "textbox", f);
      AddMutableProp(g, PROP_DRAGCOLOR, "textbox", f);
      AddMutableProp(g, PROP_FONT, "textbox", f);
      AddMutableProp(g, PROP_LINEHEIGHT, "textbox", f);
      AddMutableProp(g, PROP_LETTERSPACING, "textbox", f);
      AddMutableProp(g, PROP_VALUEF, "textbox", f);
      AddMutableProp(g, PROP_RANGE, "textbox", f);
      AddMutableProp(g, PROP_UV, "textbox", f);
      AddMutableProp(g, PROP_ASSET, "textbox", f);
      AddMutableProp(g, PROP_OUTLINE, "textbox", f);
      break;
    }
  }

  //AddMutableProp(g, PROP_CONTEXTMENU, "combobox");
  AddMutableProp(g, PROP_USERID, "textbox", f);
  AddMutableProp(g, PROP_USERINFO, "text", f, FGELEMENT_EXPANDY);

  SetProps(g, layout, element, style);
}
fgElement* EditorBase::FindProp(fgGrid& g, PROPERTIES prop)
{
  size_t len = g->GetNumItems();
  for(size_t i = 17; i < len; ++i)
  {
    fgElement* e = g.GetRow(i)->GetItem(1);
    if(e->userid == prop)
      return e;
  }
  return 0;
}

void EditorBase::SetProps(fgGrid& g, fgClassLayout* layout, fgSkinElement* element, fgStyle& style)
{
  if(element)
  {
    g.GetRow(0)->GetItem(1)->SetText(element->type);

    g.GetRow(5)->GetItem(1)->SetText(WrapWrite<CRect, fgStyle_WriteCRect>(element->transform.area, element->units).c_str());
    g.GetRow(6)->GetItem(1)->SetText(StrF("%.2g", element->transform.rotation).c_str());
    g.GetRow(7)->GetItem(1)->SetText(WrapWrite<CVec, fgStyle_WriteCVec>(element->transform.center, element->units).c_str());
    g.GetRow(13)->GetItem(1)->SetText(StrF("%zi", element->order).c_str());

    fgFlag def = fgGetTypeFlags(element->type);
    fgFlag rm = def & (~element->flags);
    fgFlag add = (~def) & element->flags;

    Str flags;
    flags.resize(fgStyle_WriteFlagsIterate(0, 0, element->type, "\n", add, 0));
    flags.resize(fgStyle_WriteFlagsIterate(flags.UnsafeString(), flags.size(), element->type, "\n", add, 0));
    size_t len = flags.size();
    flags.resize(len + fgStyle_WriteFlagsIterate(0, 0, element->type, "\n", add, 0));
    fgStyle_WriteFlagsIterate(flags.UnsafeString() + len, flags.size() - len, element->type, "\n", rm, 1);
    g.GetRow(15)->GetItem(1)->SetText(flags.c_str()[0] ? (flags.c_str() + 1) : "");
  }

  for(fgStyleMsg* m = style.styles; m != 0; m = m->next)
  {
    switch(m->msg.type)
    {
    case FG_SETSTYLE:
      g.GetRow(14)->GetItem(1)->SetText(StrF("%X", m->msg.u).c_str());
      break;
    case FG_SETCOLOR:
      if(fgElement* e = FindProp(g, PROPERTIES((FG_UINT)PROP_COLOR + m->msg.subtype)))
      {
        fgColor c = { m->msg.u };
        std::swap(c.colors[0], c.colors[3]);
        std::swap(c.colors[1], c.colors[2]);
        e->SetText(StrF("%08llX", c.color).c_str());
      }
      break;
    case FG_SETTEXT:
      if(m->msg.subtype&FGTEXTFMT_PLACEHOLDER_UTF8)
      {
        if(fgElement* e = FindProp(g, PROP_PLACEHOLDER))
        {
          FG_Msg msg = m->msg;
          msg.subtype &= 3;
          fgSendMessage(e, &msg);
        }
      }
      else if(fgElement* e = FindProp(g, PROP_TEXT))
        fgSendMessage(e, &m->msg);
      break;
    case FG_SETMARGIN:
      if(m->msg.p)
        g.GetRow(8)->GetItem(1)->SetText(WrapWrite<AbsRect, fgStyle_WriteAbsRect>(*(AbsRect*)m->msg.p, 0).c_str());
      break;
    case FG_SETPADDING:
      if(m->msg.p)
        g.GetRow(9)->GetItem(1)->SetText(WrapWrite<AbsRect, fgStyle_WriteAbsRect>(*(AbsRect*)m->msg.p, 0).c_str());
      break;
    case FG_SETDIM:
      switch(m->msg.subtype)
      {
      case FGDIM_MIN:
        g.GetRow(10)->GetItem(1)->SetText(StrF("%.2g %.2g", m->msg.f, m->msg.f2).c_str());
        break;
      case FGDIM_MAX:
        g.GetRow(11)->GetItem(1)->SetText(StrF("%.2g %.2g", m->msg.f, m->msg.f2).c_str());
        break;
      }
      break;
    case FG_SETSCALING:
      g.GetRow(12)->GetItem(1)->SetText(StrF("%.2g %.2g", m->msg.f, m->msg.f2).c_str());
      break;
    case FG_SETALPHA:
      g.GetRow(16)->GetItem(1)->SetValueF(m->msg.f);
      break;
    case FG_SETVALUE:
      switch(m->msg.subtype)
      {
      case FGVALUE_FLOAT:
        if(fgElement* e = FindProp(g, PROP_VALUEF))
          e->SetText(StrF("%.2g", m->msg.f).c_str());
        break;
      case FGVALUE_INT64:
        if(fgElement* e = FindProp(g, PROP_VALUEI))
          e->SetText(StrF("%lli", m->msg.i).c_str());
        break;
      }
      break;
    }
  }

  if(layout)
  {
    g.GetRow(1)->GetItem(1)->SetText(layout->id);
    g.GetRow(2)->GetItem(1)->SetText(layout->name);
    if(fgElement* e = FindProp(g, PROP_USERID))
      e->SetText(StrF("%u", layout->userid).c_str());
    if(fgElement* e = FindProp(g, PROP_USERINFO))
      e->SetText(StrF("%p", layout->userdata).c_str());
  }
}

void EditorBase::AddMenuControls(const char* id)
{
  fgElement* menu = fgGetID(id);
  if(menu)
    fgIterateControls(menu, [](void* p, const char* type) { auto e = (fgElement*)p; e->AddItemText(type)->userdata = (void*)type; });
}