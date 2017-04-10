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

fgLayoutEditor::fgLayoutEditor(fgLayout* layout)
{
  memset(this, 0, sizeof(fgLayoutEditor));
  fgLayoutEditor::Instance = this;

  fgSingleton()->gui->LayoutLoad(layout);

  _mainwindow = reinterpret_cast<fgWindow*>(fgGetID("Editor$mainwindow"));
  _workspace = reinterpret_cast<fgWorkspace*>(fgGetID("Editor$workspace"));
  _explorer = reinterpret_cast<fgTreeview*>(fgGetID("Editor$layout"));
  _properties = reinterpret_cast<fgGrid*>(fgGetID("Editor$properties"));
  
  /*
  { // Setup toolbar
    fgElement* mainbar = self->toolbar->box->AddItem(0);
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // New
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Open
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Save
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Save All
                                                                    // seperator
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Undo
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Redo
                                                                    // seperator
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Cut
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Copy
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Paste
    fgCreate("Button", mainbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Delete

    fgElement* viewbar = self->toolbar->box->AddItem(0);
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Show Grid
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Show Crosshair
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Hide Rulers
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Hide Cursors
                                                                    // seperator
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Snapping
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Toggle Wireframe

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