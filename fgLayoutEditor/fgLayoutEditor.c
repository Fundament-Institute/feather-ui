// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"

fgLayoutEditor* LoadLayoutEditor(fgImplementation* dll)
{
  fgLayoutEditor* editor = malloc(sizeof(fgLayoutEditor));
  memset(editor, 0, sizeof(fgLayoutEditor));
  editor->dll = dll;
  editor->root = dll->fgInitialize();
  editor->window = dll->fgTopWindow_Create("FeatherGUI Layout Editor", 0, 0);
  //editor->menu = dll->fgMenu_Create(editor->window);
  dll->fgChild_Init(&editor->workspace, 0, editor->window, 0);

  return editor;
}
void CloseLayoutEditor(fgLayoutEditor* editor)
{
  editor->dll->fgChild_Destroy(&editor->workspace);
  editor->dll->fgTerminate(editor->root);
  free(editor);
}