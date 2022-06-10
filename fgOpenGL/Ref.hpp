// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__REF_H
#define GL__REF_H

#include "GLError.hpp"

namespace GL {
  template<void (*BINDRESET)(GLenum target)> struct BindRef
  {
    constexpr BindRef() = delete;
    constexpr BindRef(GLenum target) noexcept : _target(target) {}
    constexpr BindRef(BindRef&& right) noexcept : _target(right._target) { right._target = 0; }
    BindRef(const BindRef&) = delete;
    constexpr BindRef& operator=(BindRef&& right) noexcept
    {
      _target       = right._target;
      right._target = 0;
      return *this;
    }
    BindRef& operator=(const BindRef&) = delete;

    GLenum _target;

    ~BindRef()
    {
      if(_target != 0)
        BINDRESET(_target);
    }
  };

  template<bool (*IS)(GLuint), void (*DESTROY)(GLuint)> struct Ref
  {
    constexpr Ref() noexcept : _ref(0) {}
    explicit constexpr Ref(GLuint ref) noexcept : _ref(ref) { }
    explicit constexpr Ref(void* ref) noexcept : Ref(static_cast<GLuint>(reinterpret_cast<size_t>(ref))) {}
    constexpr Ref(Ref&& right) noexcept : _ref(right.release()) {}
    constexpr Ref(const Ref&) = delete;
    constexpr ~Ref() noexcept { reset().has_value(); }
    inline constexpr GLuint release() noexcept
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

    Ref& operator=(Ref&& right) noexcept
    {
      reset().has_value();
      _ref = right.release();
      return *this;
    }
    Ref& operator=(const Ref&) = delete;
    bool constexpr operator==(const Ref& right) const noexcept { return _ref == right._ref; }
    bool constexpr operator!=(const Ref& right) const noexcept { return _ref != right._ref; }
    inline operator GLuint() const noexcept { return _ref; }
    inline operator Ref&() & noexcept { return *this; }
    inline operator const Ref&() const& noexcept { return *this; }
    inline operator Ref&&() && noexcept { return *this; }
    inline operator const Ref&&() const&& noexcept { return *this; }

    constexpr void swap(Ref& right) noexcept { std::swap(_ref, right._ref); }

  protected:
    GLuint _ref;
  };
}

#endif
