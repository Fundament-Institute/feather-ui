// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/root.h"
#include "feather/skin.h"
#include "skin.h"
#include <unordered_map>

extern "C" fgBackend* fgDirect2D(struct FG__ROOT* root);

template<class T>
fgCalcNode MakeCalc(T t)
{
  fgCalcNode node;
  if constexpr(std::is_floating_point<T>::value)
  {
    node.value.d = t;
    node.type = FG_CALC_FLOAT;
  }
  else if constexpr(std::is_integral<T>::value)
  {
    node.value.i = t;
    node.type = FG_CALC_INT;
  }
  else if constexpr(std::is_same<T, const char*>::value)
  {
    node.value.s = t;
    node.type = FG_CALC_STRING;
  }
  else if constexpr(std::is_same<std::remove_cvref<T>, std::string>::value)
  {
    node.value.s = t.c_str();
    node.type = FG_CALC_STRING;
  }
  else
    static_assert(false, "Invalid type");
  return node;
}

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

  Skin skin = {
    "root",
    0,
    {},
      {
      Skin {
        "window",
        &data,
        {
          { "behavior", { MakeCalc("window") } },
        },
        {
          Skin{ "titlebar", &data.o["title"], { { "behavior", { MakeCalc("text") } } } },
          Skin{ "button", &data.o["button"], { { "behavior", { MakeCalc("button") } } } },
        },
      }
    }
  };

  skin.Eval();
  fgSkin def = {
    kh_init_outline(),
    &skin.node,
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

  fgSkinGenerator(&root, &def, root.backend->scale, root.backend->dpi);

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