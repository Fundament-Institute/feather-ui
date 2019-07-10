// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/root.h"
#include "feather/skin.h"
#include "field.h"

extern "C" fgBackend* fgDirect2D(struct FG__ROOT* root);

int main(int argc, char** argv)
{
  Field data = {
    { "title", Field("Feather Example", "title")},
    { "image", Field("", "image")},
    { "color", Field(0xFF000000, "color")},
    { "button", 
        Field {
          { "text", Field("Push me", "text"), },
          { "onclick", Field([]() {}), },
        }},
  };
  
  fgSkin skin = {
    kh_init_outline(),
  };

  fgRoot root;
  fgInitialize(&root);
  root.backend = fgDirect2D(&root);
  root.getfield = &Field::GetField;
  root.getindex = &Field::GetIndex;
  root.setdata = &Field::SetData;
  root.setrange = &Field::SetRange;
  root.data = &data;
  root.log = stdout;

  fgSkinGenerator(&root, &skin, root.backend->scale, root.backend->dpi);

  while(fgProcessMessages(&root) == 0);

  fgTerminate(&root);

  return 0;
}

#ifdef FG_PLATFORM_WIN32
struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0, (char**)hInstance);
}
#endif