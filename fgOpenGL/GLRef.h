// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__REF_H
#define GL__REF_H

#include "GLError.h"

namespace GL {
  template<void (*BINDRESET)(GLenum target)> struct GLBindRef
  {
    constexpr GLBindRef() = delete;
    constexpr GLBindRef(GLenum target) noexcept : _target(target) {}
    constexpr GLBindRef(GLBindRef&& right) noexcept : _target(right._target) { right._target = 0; }
    GLBindRef(const GLBindRef&) = delete;
    constexpr GLBindRef& operator=(GLBindRef&& right) noexcept
    {
      _target       = right._target;
      right._target = 0;
      return *this;
    }
    GLBindRef& operator=(const GLBindRef&) = delete;

    GLenum _target;

    ~GLBindRef()
    {
      if(_target != 0)
        BINDRESET(_target);
    }
  };

  template<bool (*IS)(GLuint), void (*DESTROY)(GLuint)> struct GLRef
  {
    constexpr GLRef() noexcept : _ref(0) {}
    explicit constexpr GLRef(GLuint ref) noexcept : _ref(ref) { assert(is_valid()); }
    explicit constexpr GLRef(void* ref) noexcept : GLRef(static_cast<GLuint>(reinterpret_cast<size_t>(ref))) {}
    constexpr GLRef(GLRef&& right) noexcept : _ref(std::move(right).take()) {}
    constexpr GLRef(const GLRef&) = delete;
    constexpr ~GLRef() noexcept { reset().has_value(); }
    inline constexpr GLuint take() && noexcept
    {
      auto ref = _ref;
      _ref     = 0;
      return ref;
    }
    bool constexpr is_valid() const noexcept { return IS(_ref) == GL_TRUE; }
    GLExpected<void> constexpr reset() noexcept
    {
      if(is_valid())
      {
        DESTROY(_ref);
        _ref = 0;
        GL_ERROR("Deletion failure");
      }

      return {};
    }

    GLRef& operator=(GLRef&& right) noexcept
    {
      reset().has_value();
      _ref = std::move(right).take();
    }
    GLRef& operator=(const GLRef&) = delete;
    bool constexpr operator==(const GLRef& right) const noexcept { return _ref == right._ref; }
    bool constexpr operator!=(const GLRef& right) const noexcept { return _ref != right._ref; }
    inline operator GLuint() const noexcept { return _ref; }
    inline operator GLRef&() & noexcept { return *this; }
    inline operator const GLRef&() const& noexcept { return *this; }
    inline operator GLRef&&() && noexcept { return *this; }
    inline operator const GLRef&&() const&& noexcept { return *this; }

    constexpr void swap(GLRef& right) noexcept { std::swap(_ref, right._ref); }

  protected:
    GLuint _ref;
  };
}

#endif
