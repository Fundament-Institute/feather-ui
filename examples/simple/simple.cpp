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
          { "title", { MakeCalc("feather test") } },
          { "left-abs", { MakeCalc(0.0f) } },
          { "top-abs", { MakeCalc(0.0f) } },
          { "right-abs", { MakeCalc(300.0f) } },
          { "bottom-abs", { MakeCalc(200.0f) } },
        },
        {
          Skin{ "bg", &data, 
        {
          { "behavior", { MakeCalc("box") } },
          { "left-abs", { MakeCalc(0.0f) } },
          { "top-abs", { MakeCalc(0.0f) } },
          { "right-abs", { MakeCalc(300.0f) } },
          { "bottom-abs", { MakeCalc(200.0f) } },
          { "fillcolor", { MakeCalc(0xFFCCCCCC) } },
          { "bordercolor", { MakeCalc(0xFF000000) } },
          { "border", { MakeCalc(1.0f) } },
        }
          },
          /*Skin{ "titlebar", &data.o["title"], { { "behavior", { MakeCalc("text") } } } },*/
          Skin{ "button", &data.o["button"], 
            { 
              { "behavior", { MakeCalc("button") } },
              { "left-abs", { MakeCalc(50.0f) } },
              { "top-abs", { MakeCalc(50.0f) } },
              { "right-abs", { MakeCalc(140.0f) } },
              { "bottom-abs", { MakeCalc(74.0f) } },
            },
            {
              Skin{ "bg", &data,
                {
                  { "behavior", { MakeCalc("box") } },
                  { "left-abs", { MakeCalc(0.0f) } },
                  { "top-abs", { MakeCalc(0.0f) } },
                  { "right-abs", { MakeCalc(90.0f) } },
                  { "bottom-abs", { MakeCalc(24.0f) } },
                  { "fillcolor", { MakeCalc(0xFF5D6B99) } },
                  { "bordercolor", { MakeCalc(0xFFCCCFF0) } },
                  { "border", { MakeCalc(1.0f) } },
                }
              },
              Skin{ "caption", &data,
                {
                  { "behavior", { MakeCalc("text") } },
                  { "left-abs", { MakeCalc(10.0f) } },
                  { "top-abs", { MakeCalc(4.0f) } },
                  { "right-abs", { MakeCalc(80.0f) } },
                  { "bottom-abs", { MakeCalc(20.0f) } },
                  { "fillcolor", { MakeCalc(0xFFFFFFFF) } },
                  { "text", { MakeCalc("Push Me") } },
                  { "font-weight", { MakeCalc(400) } },
                  { "fontsize", { MakeCalc(12.0f) } },
                  { "lineheight", { MakeCalc(16.0f) } },
                  { "font-family", { MakeCalc("Verdana") } },
                }
              },
            },
          }
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
  for(;;)
    fgProcessMessages(&root);

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