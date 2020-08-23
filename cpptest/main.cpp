#include "backend.h"
#include <stdio.h>
#include <stdarg.h>

#define BACKEND fgOpenGL
#define TEST(x)                      \
  if(!(x))                           \
  {                                  \
    printf("Failed test: %s\n", #x); \
  }

extern "C" FG_Backend* BACKEND(void* root, FG_Log log, FG_Behavior behavior);

const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "DEBUG: " };

void FakeLog(void* root, FG_Level level, const char* f, ...)
{
  if(level >= 0)
    printf(LEVELS[level]);

  va_list args;
  va_start(args, f);
  vprintf(f, args);
  va_end(args);
  printf("\n");
}

struct MockElement : FG_Element
{
  FG_Asset* image;
  FG_Font* font;
  void* layout;
  uint64_t flags;
  bool close;
};

FG_Result behavior(FG_Element* element, void* w, void* ui, FG_Msg* m)
{
  MockElement& e = *static_cast<MockElement*>(element);
  if(m->kind == FG_Kind_DRAW)
  {
    FG_Backend* b = *(FG_Backend**)ui;
    auto r        = FG_Rect{ 50.f, 100.f, 200.f, 300.f };
    auto c        = FG_Rect{ 0.f, 4.f, 8.f, 12.f };
    (*b->drawRect)(b, w, &r, &c, FG_Color{ 0xFF0000FF }, 5.f, FG_Color{ 0xFF00FFFF }, 0.f, nullptr, 0.f, 0.f);

    auto r2 = FG_Rect{ 350.f, 100.f, 500.f, 300.f };
    auto c2 = FG_Vec{ 0.f, 3.f };
    (*b->drawCircle)(b, w, &r2, &c2, FG_Color{ 0xFF0000FF }, 5.f, FG_Color{ 0xFF00FFFF }, 0.f, 10.f, 2.f, nullptr, 0.f);

    auto r3 = FG_Rect{ 150.f, 300.f, 300.f, 500.f };
    auto c3 = FG_Rect{ 0.f, 4.f, 8.f, 0.5f };
    (*b->drawTriangle)(b, w, &r3, &c3, FG_Color{ 0xFF0000FF }, 5.f, FG_Color{ 0xFF00FFFF }, 0.f, nullptr, 0.45f, 0.f);

    auto r4 = FG_Rect{ 300.f, 300.f, 620.f, 580.f };
    (*b->drawAsset)(b, w, e.image, &r4, nullptr, FG_Color{ 0xFFFFFFFF }, 0.f, 0.f, 0.f);

    auto r5 = FG_Rect{ 200.f, 10.f, 550.f, 40.f };
    (*b->drawText)(b, w, e.font, e.layout, &r5, FG_Color{ 0xFFFFFFFF }, 0.f, 0.f, 0.f);
    return FG_Result{ 0 };
  }
  if(m->kind == FG_Kind_GETWINDOWFLAGS)
  {
    return e.close ? FG_Result{ FG_Window_CLOSED } : FG_Result{ 0 };
  }
  if(m->kind == FG_Kind_SETWINDOWFLAGS)
  {
    e.close = e.close || ((m->setWindowFlags.flags and FG_Window_CLOSED) != 0);
  }
  if(m->kind == FG_Kind_KEYDOWN || m->kind == FG_Kind_MOUSEDOWN)
  {
    e.close = true;
  }
  return FG_Result{ -1 };
}

int main(int argc, char* argv[])
{
  FG_Backend* ui;
  auto b        = BACKEND(&ui, FakeLog, behavior);
  ui            = b;
  auto textrect = FG_Rect{ 0.f, 0.f, 1000.f, 700.f };
  auto e        = MockElement{};

  e.flags  = FG_Window_RESIZABLE;
  e.image  = (*b->createAsset)(b, "../tests/example.png", 0, FG_Format_PNG);
  e.font   = (*b->createFont)(b, "Arial", 700, false, 16, FG_Vec{ 96.f, 96.f }, FG_AntiAliasing_AA);
  e.layout = (*b->fontLayout)(b, e.font, "Example Text!", &textrect, 16.f, 0.f, FG_BreakStyle_NONE, nullptr);
  e.close  = false;

  auto pos = FG_Vec{ 200.f, 100.f };
  auto dim = FG_Vec{ 800.f, 600.f };
  auto w   = (*b->createWindow)(b, &e, nullptr, &pos, &dim, "Feather Test", e.flags, nullptr);

  TEST((*b->setCursor)(b, w, FG_Cursor_CROSS) == 0);
  TEST((*b->requestAnimationFrame)(b, w, 0) == 0);

  TEST((*b->setWindow)(b, w, nullptr, nullptr, nullptr, nullptr, nullptr, FG_Window_RESIZABLE) == 0);
  TEST((*b->setWindow)(b, w, &e, nullptr, nullptr, nullptr, "Feather Test Changed", FG_Window_RESIZABLE) == 0);

  TEST((*b->processMessages)(b) != 0);

  const char TEST_TEXT[] = "testtext";

  TEST(w != nullptr);
  TEST((*b->processMessages)(b) != 0);
  TEST((*b->clearClipboard)(b, w, FG_Clipboard_ALL) == 0);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_TEXT) == false);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_WAVE) == false);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_ALL) == false);
  TEST((*b->putClipboard)(b, w, FG_Clipboard_TEXT, TEST_TEXT, 9) == 0);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_TEXT) == true);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_WAVE) == false);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_ALL) == true);

  char hold[10];

  TEST((*b->getClipboard)(b, w, FG_Clipboard_TEXT, hold, 10) == 9)
  for(int i = 0; i < 8; ++i)
  {
    TEST(TEST_TEXT[i] == hold[i]);
  }

  TEST((*b->getClipboard)(b, w, FG_Clipboard_WAVE, hold, 10) == 0)

  while((*b->processMessages)(b) != 0 && e.close == false)
  {
    (*b->requestAnimationFrame)(b, w, 0);
  }

  TEST((*b->destroyWindow)(b, w) == 0)
  TEST((*b->destroyAsset)(b, e.image) == 0)
  TEST((*b->destroyLayout)(b, e.layout) == 0)
  TEST((*b->destroyFont)(b, e.font) == 0)
  (*b->destroy)(b);
}