// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__WINDOW_H
#define GL__WINDOW_H

#include "Context.h"

namespace GL {
  class Backend;
  struct Asset;

  struct Window : Context
  {
    Window(Backend* backend, GLFWmonitor* display, FG_MsgReceiver* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags,
           const char* caption);
    virtual ~Window();
    size_t SetKey(uint8_t keycode, bool down, bool held, unsigned long time);
    size_t SetChar(int key, unsigned long time);
    size_t SetMouse(FG_Vec& points, FG_Kind type, unsigned char button, size_t wparam, unsigned long time);
    virtual void DirtyRect(const FG_Rect* rect) override;
    static uint8_t GetModKeys(int mods);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CloseCallback(GLFWwindow* window);
    static void CharCallback(GLFWwindow* window, unsigned int key);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void MousePosCallback(GLFWwindow* window, double x, double y);
    static void EnterCallback(GLFWwindow* window, int entered);
    static void ScrollCallback(GLFWwindow* window, double xdelta, double ydelta);
    static void DropCallback(GLFWwindow* window, int count, const char* paths[]);
    static void FocusCallback(GLFWwindow* window, int focused);
    static void SizeCallback(GLFWwindow* window, int width, int height);
    static void RefreshCallback(GLFWwindow* window);
    
    uint64_t _flags;
    Window* _next; // GLFW doesn't let us detect when it destroys a window so we have to do it ourselves.
    Window* _prev;

    static uint8_t KeyMap[512];
    static void FillKeyMap();
  };
}

#endif
