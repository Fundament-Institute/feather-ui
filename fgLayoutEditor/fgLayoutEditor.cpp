// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

void fgLayoutEditor_Init(fgLayoutEditor* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  memset(self, 0, sizeof(fgLayoutEditor));
  self->window = reinterpret_cast<fgWindow*>(fgCreate("Window", parent, next, name, flags, transform, units));
  self->main = reinterpret_cast<fgBox*>(fgCreate("Box", *self->window, 0, "fgLayoutEditor$main", FGBOX_TILEY, &fgTransform_EMPTY, 0));
  self->menu = reinterpret_cast<fgMenu*>(fgCreate("Menu", *self->window, 0, "fgLayoutEditor$menu", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->toolbar = reinterpret_cast<fgToolbar*>(fgCreate("Toolbar", *self->window, 0, "fgLayoutEditor$toolbar", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->explorer = reinterpret_cast<fgTreeview*>(fgCreate("Treeview", *self->window, 0, "fgLayoutEditor$explorer", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->properties = reinterpret_cast<fgGrid*>(fgCreate("Grid", *self->window, 0, "fgLayoutEditor$properties", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->workspace = reinterpret_cast<fgWorkspace*>(fgCreate("Workspace", *self->window, 0, "fgLayoutEditor$workspace", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
  self->context = reinterpret_cast<fgMenu*>(fgCreate("Submenu", *self->window, 0, "fgLayoutEditor$context", FGELEMENT_USEDEFAULTS, &fgTransform_EMPTY, 0));
}

void fgLayoutEditor_Destroy(fgLayoutEditor* self)
{
  VirtualFreeChild(*self->window);
}