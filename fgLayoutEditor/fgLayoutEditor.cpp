// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

void fgLayoutEditor_Init(fgLayoutEditor* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  memset(self, 0, sizeof(fgLayoutEditor));
  self->window = reinterpret_cast<fgWindow*>(fgCreate("Window", parent, next, name, flags, transform, units));
  self->main = reinterpret_cast<fgBox*>(fgCreate("Box", *self->window, 0, "fgLayoutEditor$main", FGBOX_TILEY, &fgTransform_EMPTY, 0));
  self->menu = reinterpret_cast<fgMenu*>(fgCreate("Menu", *self->main, 0, "fgLayoutEditor$menu", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->toolbar = reinterpret_cast<fgToolbar*>(fgCreate("Toolbar", *self->main, 0, "fgLayoutEditor$toolbar", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->workspace = reinterpret_cast<fgWorkspace*>(fgCreate("Workspace", *self->main, 0, "fgLayoutEditor$workspace", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->explorer = reinterpret_cast<fgTreeview*>(fgCreate("Treeview", *self->workspace, 0, "fgLayoutEditor$explorer", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->properties = reinterpret_cast<fgGrid*>(fgCreate("Grid", *self->workspace, 0, "fgLayoutEditor$properties", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->context = reinterpret_cast<fgMenu*>(fgCreate("Submenu", *self->window, 0, "fgLayoutEditor$context", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  
  { // Setup menus
    fgElement* file = fgCreate("Submenu", self->menu->box->AddItemText("File"), 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
    fgElement* edit = fgCreate("Submenu", self->menu->box->AddItemText("Edit"), 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
    fgElement* view = fgCreate("Submenu", self->menu->box->AddItemText("View"), 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
    fgElement* help = fgCreate("Submenu", self->menu->box->AddItemText("Help"), 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);

    file->AddItemText("New");
    file->AddItemText("Open");
    file->AddItemText("Open Recent");
    file->AddItemText("Save");
    file->AddItemText("Save As...");
    file->AddItemText("Save All");
    file->AddItemText(0);
    file->AddItemText("Close");
    file->AddItemText("Close All");
    file->AddItemText(0);
    file->AddItemText("Exit");

    edit->AddItemText("Insert");
    edit->AddItemText("Delete");
    edit->AddItemText("Move");
    edit->AddItemText("Select All");
    edit->AddItemText(0);
    edit->AddItemText("Cut");
    edit->AddItemText("Copy");
    edit->AddItemText("Paste");
    edit->AddItemText(0);
    edit->AddItemText("Undo");
    edit->AddItemText("Redo");
    edit->AddItemText(0);

    view->AddItemText("Show Grid");
    view->AddItemText("Show Crosshair");
    view->AddItemText("Hide Rulers");
    view->AddItemText("Hide Cursors");
    view->AddItemText(0);
    view->AddItemText("Snap to Grid");
    view->AddItemText("Snap Near");
    view->AddItemText(0);
    view->AddItemText("Toggle Wireframe");

    help->AddItemText("Manual");
    help->AddItemText("Documentation");
    help->AddItemText(0);
    help->AddItemText("About FeatherGUI Layout Editor");
  }

  { // Setup toolbar
    fgElement* mainbar = self->toolbar->list->AddItem(0);
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

    fgElement* viewbar = self->toolbar->list->AddItem(0);
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Show Grid
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Show Crosshair
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Hide Rulers
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Hide Cursors
                                                                    // seperator
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Snapping
    fgCreate("Button", viewbar, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0); // Toggle Wireframe

    fgElement* insertbar = self->toolbar->list->AddItem(0); // Bar with buttons for each element
    fgIterateControls(insertbar, [](void* p, const char* s) { fgElement* e = (fgElement*)p; e->AddItemText(s); });
  }


}

void fgLayoutEditor_Destroy(fgLayoutEditor* self)
{
  VirtualFreeChild(*self->window);
}