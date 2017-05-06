// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

using namespace bss;

fgLayoutEditor* fgLayoutEditor::Instance = 0;

size_t fgLayoutEditor::_inject(fgRoot* self, const FG_Msg* msg)
{
  if(msg->type == FG_KEYDOWN)
  {
    if(msg->keycode == FG_KEY_F11)
    {
      if(fgDebug_Get() != 0 && !(fgDebug_Get()->tabs.control.element.flags&FGELEMENT_HIDDEN))
        fgDebug_Hide();
      else
        fgDebug_Show(0, 1);
    }
  }
  return fgRoot_DefaultInject(self, msg);
}

size_t fgLayoutEditor::_propertyMessage(fgText* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEUP:
    if(self->element.parent && self->element.parent->parent && self->element.parent->parent->parent)
    {
      AbsRect rect;
      fgElement_RelativeTo(&self->element, self->element.parent->parent->parent, &rect);
      CRect area = { rect.left, 0, rect.top, 0, rect.right, 0, rect.bottom, 0 };
      fgElement* textbox = fgLayoutEditor::Instance->EditBox;
      if(textbox->userdata)
        ((fgElement*)textbox->userdata)->SetFlag(FGELEMENT_HIDDEN, false);
      textbox->SetArea(area);
      textbox->SetText(self->element.GetText());
      textbox->SetFlag(FGELEMENT_HIDDEN, false);
      textbox->userdata = self;
      textbox->userid = self->element.userid;
      self->element.SetFlag(FGELEMENT_HIDDEN, true);
    }
    break;
  }

  return fgText_Message(self, msg);
}

void fgLayoutEditor::_addprop(fgGrid& g, const char* name, const char* type, FG_UINT userid)
{
  static const fgTransform TF1 = { { 0,0,0,0,0,0.5,20,0 }, 0,{ 0,0,0,0 } };
  static const fgTransform TF2 = { {0,0.5,0,0,0,1,20,0}, 0, {0,0,0,0} };
  fgGridRow* row = g.InsertRow();
  fgCreate("Text", *row, 0, 0, FGELEMENT_EXPANDY, &fgTransform_EMPTY, 0)->SetText(name);
  fgElement* prop = fgCreate(type, *row, 0, 0, FGELEMENT_EXPANDY, &fgTransform_EMPTY, 0);
  prop->userid = userid;
  prop->message = (fgMessage)&fgLayoutEditor::_propertyMessage;
}

void fgLayoutEditor::_addmutableprop(fgGrid& g, MUTABLE_PROPERTIES id, const char* type)
{
  const char* names[PROP_TOTALPLUSONE - 1] = { "UserID", "UserInfo", "Text", "Placeholder", "Font", "Line Height", "Letter Spacing", "Color",
    "Color:Placeholder", "Color:Cursor", "Color:Select", "Color:Hover", "Color:Drag","Color:Edge", "Color:Divider", "Color:ColumnDivider", "Value",
    "Value", "Range", "UV", "Asset", "Outline", "Splitter", "Context Menu" };

  _addprop(g, names[id], type, id);
}

void fgLayoutEditor::_clearprops(fgGrid& g, fgClassLayout& layout)
{
  static Trie<uint16_t, true> t(30, "element", "control", "scrollbar", "box", "list", "grid", "resource", "text", "button", "window", "checkbox", "radiobutton",
    "progressbar", "slider", "textbox", "treeview", "treeitem", "listitem", "curve", "dropdown", "tabcontrol", "menu", "submenu",
    "menuitem", "gridrow", "workspace", "toolbar", "toolgroup", "combobox", "debug");

  for(size_t i = 0; i < 17; ++i)
    g.GetRow(i)->GetItem(1)->SetText("");

  while(g.GetNumRows() > 17)
    g.RemoveRow(g.GetNumRows() - 1);
   
  switch(t[layout.layout.type])
  {
  case 0: // element
  default:
    break;
  case 11: // radiobutton
  case 10: // checkbox
    _addmutableprop(g, PROP_VALUEI, "text");
  case 7: // text
  case 8: // button
  case 9: // window
  case 28: // combobox
    _addmutableprop(g, PROP_TEXT, "text");
    _addmutableprop(g, PROP_COLOR, "text");
    _addmutableprop(g, PROP_FONT, "text");
    _addmutableprop(g, PROP_LINEHEIGHT, "text");
    _addmutableprop(g, PROP_LETTERSPACING, "text");
    break;
  case 12: // progressbar
    _addmutableprop(g, PROP_VALUEF, "text");
    _addmutableprop(g, PROP_TEXT, "text");
    _addmutableprop(g, PROP_COLOR, "text");
    _addmutableprop(g, PROP_FONT, "text");
    _addmutableprop(g, PROP_LINEHEIGHT, "text");
    _addmutableprop(g, PROP_LETTERSPACING, "text");
    break;
  case 14: // textbox
    _addmutableprop(g, PROP_TEXT, "text");
    _addmutableprop(g, PROP_PLACEHOLDER, "text");
    _addmutableprop(g, PROP_COLOR, "text");
    _addmutableprop(g, PROP_PLACECOLOR, "text");
    _addmutableprop(g, PROP_CURSORCOLOR, "text");
    _addmutableprop(g, PROP_SELECTCOLOR, "text");
    _addmutableprop(g, PROP_FONT, "text");
    _addmutableprop(g, PROP_LINEHEIGHT, "text");
    _addmutableprop(g, PROP_LETTERSPACING, "text");
    break;
  case 19: // dropdown
    _addmutableprop(g, PROP_SELECTCOLOR, "text");
    _addmutableprop(g, PROP_HOVERCOLOR, "text");
    break;
  case 4: // list
    _addmutableprop(g, PROP_DRAGCOLOR, "text");
    _addmutableprop(g, PROP_VALUEF, "text");
    break;
  case 5: // grid
    _addmutableprop(g, PROP_DRAGCOLOR, "text");
    _addmutableprop(g, PROP_SPLITTER, "text");
    break;
  case 23: // menuitem:
    _addmutableprop(g, PROP_TEXT, "text");
    break;
  case 6: // resource
    _addmutableprop(g, PROP_UV, "text");
    _addmutableprop(g, PROP_ASSET, "text");
    _addmutableprop(g, PROP_OUTLINE, "text");
    _addmutableprop(g, PROP_COLOR, "text");
    _addmutableprop(g, PROP_EDGECOLOR, "text");
    break;
  case 18: // curve
    _addmutableprop(g, PROP_COLOR, "text");
    _addmutableprop(g, PROP_VALUEF, "text");
    break;
  case 13: // slider
    _addmutableprop(g, PROP_VALUEF, "text");
    _addmutableprop(g, PROP_RANGE, "text");
    break;
  }

  //_addmutableprop(g, PROP_CONTEXTMENU, "combobox");
  _addmutableprop(g, PROP_USERID, "text");
  _addmutableprop(g, PROP_USERINFO, "text");

  _setprops(g, layout);
}

fgElement* fgLayoutEditor::FindProp(fgGrid& g, MUTABLE_PROPERTIES prop)
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

template<typename T, size_t (*F)(char*, size_t, const T*, short)>
Str WrapWrite(const T& v, short u)
{
  Str s;
  s.resize(F(0, 0, &v, u));
  s.resize(F(s.UnsafeString(), s.size(), &v, u));
  return s;
}
void fgLayoutEditor::_setprops(fgGrid& g, fgClassLayout& layout)
{
  g.GetRow(0)->GetItem(1)->SetText(layout.layout.type);
  g.GetRow(1)->GetItem(1)->SetText(layout.id);
  g.GetRow(2)->GetItem(1)->SetText(layout.name);

  g.GetRow(5)->GetItem(1)->SetText(WrapWrite<CRect, fgStyle_WriteCRect>(layout.layout.transform.area, layout.layout.units).c_str());
  g.GetRow(6)->GetItem(1)->SetText(StrF("%.2g", layout.layout.transform.rotation).c_str());
  g.GetRow(7)->GetItem(1)->SetText(WrapWrite<CVec, fgStyle_WriteCVec>(layout.layout.transform.center, layout.layout.units).c_str());
  g.GetRow(14)->GetItem(1)->SetText(StrF("%zi", layout.layout.order).c_str());
  
  fgFlag def = fgGetTypeFlags(layout.layout.type);
  fgFlag rm = def & (~layout.layout.flags);
  fgFlag add = (~def) & layout.layout.flags;

  Str flags;
  flags.resize(fgStyle_WriteFlagsIterate(0, 0, layout.layout.type, "\n", add, 0));
  flags.resize(fgStyle_WriteFlagsIterate(flags.UnsafeString(), flags.size(), layout.layout.type, "\n", add, 0));
  size_t len = flags.size();
  flags.resize(len + fgStyle_WriteFlagsIterate(0, 0, layout.layout.type, "\n", add, 0));
  fgStyle_WriteFlagsIterate(flags.UnsafeString() + len, flags.size() - len, layout.layout.type, "\n", rm, 1);
  g.GetRow(15)->GetItem(1)->SetText(flags.c_str()[0] ? (flags.c_str() + 1) : "");

  for(fgStyleMsg* m = layout.layout.style.styles; m != 0; m = m->next)
  {
    switch(m->msg.type)
    {
    case FG_SETSTYLE:
      g.GetRow(13)->GetItem(1)->SetText(StrF("%X", m->msg.u).c_str());
      break;
    case FG_SETCOLOR:
      if(fgElement* e = FindProp(g, MUTABLE_PROPERTIES((FG_UINT)PROP_COLOR + m->msg.subtype)))
        e->SetText(StrF("%X", m->msg.u).c_str());
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
  if(fgElement* e = FindProp(g, PROP_USERID))
    e->SetText(StrF("%u", layout.userid).c_str());
  if(fgElement* e = FindProp(g, PROP_USERINFO))
    e->SetText(StrF("%p", layout.userdata).c_str());
}

fgLayoutEditor::fgLayoutEditor(fgLayout* layout, EditorSettings& settings) : _settings(settings)
{
  memset(this, 0, sizeof(fgLayoutEditor));
  fgLayoutEditor::Instance = this;

  fgSingleton()->gui->LayoutLoad(layout);

  _mainwindow = reinterpret_cast<fgWindow*>(fgGetID("Editor$mainwindow"));
  _workspace = reinterpret_cast<fgWorkspace*>(fgGetID("Editor$workspace"));
  _explorer = reinterpret_cast<fgTreeview*>(fgGetID("Editor$layout"));
  _properties = reinterpret_cast<fgGrid*>(fgGetID("Editor$properties"));
  fgElement* menu = fgGetID("Editor$main");
  
  if(_workspace)
    _workspace->scroll->LayoutLoad(layout);

  if(_properties)
  {
    _addprop(*_properties, "(Type)");
    _addprop(*_properties, "ID");
    _addprop(*_properties, "Name");
    _addprop(*_properties, "Skin", "combobox");
    _addprop(*_properties, "Transform");
    _addprop(*_properties, "  Area");
    _addprop(*_properties, "  Rotation");
    _addprop(*_properties, "  Center");
    _addprop(*_properties, "Margin");
    _addprop(*_properties, "Padding");
    _addprop(*_properties, "Min Dim");
    _addprop(*_properties, "Max Dim");
    _addprop(*_properties, "Scaling");
    _addprop(*_properties, "Style");
    _addprop(*_properties, "Order");
    _addprop(*_properties, "Flags");
    _addprop(*_properties, "Alpha", "Slider");

  }
  /*
  { // Setup toolbar
    fgElement* mainbar = self->toolbar->box->AddItem(0);
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // New
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Open
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Save
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Save All
                                                                    // seperator
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Undo
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Redo
                                                                    // seperator
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Cut
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Copy
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Paste
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Delete

    fgElement* viewbar = self->toolbar->box->AddItem(0);
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Show Grid
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Show Crosshair
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Hide Rulers
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Hide Cursors
                                                                    // seperator
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Snapping
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Toggle Wireframe

    fgElement* insertbar = self->toolbar->box->AddItem(0); // Bar with buttons for each element
    fgIterateControls(insertbar, [](void* p, const char* s) { fgElement* e = (fgElement*)p; e->AddItemText(s); });
  }*/

  fgSetInjectFunc(_inject);
}

fgLayoutEditor::~fgLayoutEditor()
{
  fgElement* window = fgGetID("Editor$mainwindow");
  if(window)
    VirtualFreeChild(window);
}
void fgLayoutEditor::_openlayout(fgElement* root, const fgVectorClassLayout& layout)
{
  for(size_t i = 0; i < layout.l; ++i)
  {
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = layout.p + i;
    fgElement_AddListener(item, FG_GOTFOCUS, &ExplorerOnFocus);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(layout.p[i].layout.type);
    _openlayout(item, layout.p[i].children);
  }
}
void fgLayoutEditor::OpenLayout(fgLayout* layout)
{
  if(_explorer)
  {
    fgElement_Clear(*_explorer);

    fgCreate("Text", *_explorer, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText("root");
    _openlayout(*_explorer, layout->layout);
  }
}
void fgLayoutEditor::MenuFile(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  if(m->e)
  {
    switch(m->e->userid)
    {
    case 8:
      if(Instance->_mainwindow)
        VirtualFreeChild(*Instance->_mainwindow);
    }
  }
}
void fgLayoutEditor::MenuRecent(struct _FG_ELEMENT*, const FG_Msg*)
{}
void fgLayoutEditor::MenuEdit(struct _FG_ELEMENT*, const FG_Msg*)
{

}
void fgLayoutEditor::MenuView(struct _FG_ELEMENT*, const FG_Msg*)
{

}
void fgLayoutEditor::MenuHelp(struct _FG_ELEMENT*, const FG_Msg*)
{

}
void fgLayoutEditor::ExplorerOnFocus(struct _FG_ELEMENT* e, const FG_Msg*)
{
  fgClassLayout* layout = (fgClassLayout*)e->userdata;
  if(layout)
    Instance->_clearprops(*Instance->_properties, *layout);
}
