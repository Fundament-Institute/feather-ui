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

fgLayoutEditor::fgLayoutEditor(fgLayout* layout, EditorSettings& settings) : EditorBase(layout), _settings(settings)
{
  fgLayoutEditor::Instance = this;
  _layout.Init(this);
  _skin.Init(this);

  _mainwindow = reinterpret_cast<fgWindow*>(fgGetID("Editor$mainwindow"));
  _workspace = reinterpret_cast<fgWorkspace*>(fgGetID("Editor$workspace"));
  if(_workspace)
    (*_workspace)->message = (fgMessage)WorkspaceMessage;
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
}

void fgLayoutEditor::Destroy()
{
  _layout.Clear();
  _skin.Clear();
}
fgElement* fgLayoutEditor::LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout)
{
  fgElement* element = fgCreate(layout->layout.type, parent, next, layout->name, layout->layout.flags, (layout->layout.units == -1) ? 0 : &layout->layout.transform, layout->layout.units | FGUNIT_SNAP);
  assert(element != 0);
  if(!element)
    return 0;
  Instance->_layout.Link(element, layout);
  element->userid = layout->userid;
  fgElement_StyleToMessageArray(&layout->layout.style, 0, &element->layoutstyle);
  if(element->layoutstyle)
    fgElement_ApplyMessageArray(0, element, element->layoutstyle);

  for(FG_UINT i = 0; i < layout->children.l; ++i)
    LoadLayout(element, 0, layout->children.p + i);

  return element;
}

size_t fgLayoutEditor::WorkspaceMessage(fgWorkspace* e, const FG_Msg* m)
{
  switch(m->type)
  {
  case FG_INJECT:
    if(!fgSingleton()->GetKey(FG_KEY_MENU))
      return FG_ACCEPT;
    break;
  case FG_LAYOUTLOAD:
    fgLayout* layout = (fgLayout*)m->p;
    if(!layout)
      return 0;

    Instance->_layout.ClearLinks();
    fgElement* last = 0;
    for(FG_UINT i = 0; i < layout->layout.l; ++i)
      last = LoadLayout(*e, 0, layout->layout.p + i);
    return (size_t)last;
  }

  return fgWorkspace_Message(e, m);
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
void fgLayoutEditor::Close()
{
  if(_workspace)
    fgElement_Clear(*_workspace);
  _layout.Clear();
  _skin.Clear();
}
void fgLayoutEditor::OpenLayout(fgLayout* layout)
{
  Close();
  _layout.OpenLayout(layout);
  DisplayLayout(layout);
}
void fgLayoutEditor::DisplayLayout(fgLayout* layout)
{
  if(_workspace)
  {
    fgElement_Clear(*_workspace);
    (*_workspace)->LayoutLoad(layout);
  }
}
void fgLayoutEditor::LoadFile(const char* file)
{
  NewFile();
  fgLayout_LoadFileXML(&curlayout, file);
}
void fgLayoutEditor::SaveFile(const char* file)
{
  fgLayout_SaveFileXML(&curlayout, file);
}
void fgLayoutEditor::NewFile()
{
  fgLayout_Destroy(&curlayout);
  fgLayout_Init(&curlayout);
}