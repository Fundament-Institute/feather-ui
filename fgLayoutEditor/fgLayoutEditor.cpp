// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

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

void fgLayoutEditor::_addprop(fgGrid* e, const char* name)
{
  static const fgTransform TF1 = { { 0,0,0,0,0,0.5,20,0 }, 0,{ 0,0,0,0 } };
  static const fgTransform TF2 = { {0,0.5,0,0,0,1,20,0}, 0, {0,0,0,0} };
  fgGridRow* row = e->InsertRow();
  fgCreate("Text", *row, 0, 0, 0, &TF1, 0)->SetText(name);
  fgCreate("Text", *row, 0, 0, 0, &TF2, 0);
}

void fgLayoutEditor::_setprops(fgGrid* e, fgClassLayout& layout)
{
  e->GetRow(0)->GetItem(1)->SetText(layout.layout.type);
  e->GetRow(1)->GetItem(1)->SetText(layout.id);
  e->GetRow(2)->GetItem(1)->SetText(layout.name);
}

fgLayoutEditor::fgLayoutEditor(fgLayout* layout)
{
  memset(this, 0, sizeof(fgLayoutEditor));
  fgLayoutEditor::Instance = this;

  fgSingleton()->gui->LayoutLoad(layout);

  _mainwindow = reinterpret_cast<fgWindow*>(fgGetID("Editor$mainwindow"));
  _workspace = reinterpret_cast<fgWorkspace*>(fgGetID("Editor$workspace"));
  _explorer = reinterpret_cast<fgTreeview*>(fgGetID("Editor$layout"));
  _properties = reinterpret_cast<fgGrid*>(fgGetID("Editor$properties"));
  
  _addprop(_properties, "(Type)");
  _addprop(_properties, "ID");
  _addprop(_properties, "Name");
  _addprop(_properties, "Skin");
  _addprop(_properties, "Transform");
  _addprop(_properties, "  Area");
  _addprop(_properties, "  Rotation");
  _addprop(_properties, "  Center");
  _addprop(_properties, "Margin");
  _addprop(_properties, "Padding");
  _addprop(_properties, "Flags");
  _addprop(_properties, "Order");
  _addprop(_properties, "Style");
  _addprop(_properties, "Alpha");
  _addprop(_properties, "Text");
  _addprop(_properties, "Placeholder");
  _addprop(_properties, "Color");
  _addprop(_properties, "Font");
  _addprop(_properties, "Line Height");
  _addprop(_properties, "Letter Spacing");
  _addprop(_properties, "Value");
  _addprop(_properties, "UV");
  _addprop(_properties, "Asset");
  _addprop(_properties, "Outline");
  _addprop(_properties, "Context Menu");
  _addprop(_properties, "Min Dim");
  _addprop(_properties, "Max Dim");
  _addprop(_properties, "UserID");
  _addprop(_properties, "UserData");

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
  fgElement_Clear(*_explorer);
  
  fgCreate("Text", *_explorer, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText("root");
  _openlayout(*_explorer, layout->layout);
}
void fgLayoutEditor::MenuFile(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  fgElement* child = (fgElement*)m->p;
  if(child)
  {
    switch(child->userid)
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
    Instance->_setprops(Instance->_properties, *layout);
}
