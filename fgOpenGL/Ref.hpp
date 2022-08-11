// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__REF_H
#define GL__REF_H

#include "GLError.hpp"
#include "feather/backend.h"

namespace GL {
  enum RefType : FG_Resource
  {
    REF_TEXTURE      = ((FG_Resource)1 << (sizeof(FG_Resource) * 8 - 3)),
    REF_BUFFER       = ((FG_Resource)2 << (sizeof(FG_Resource) * 8 - 3)),
    REF_FRAMEBUFFER  = ((FG_Resource)3 << (sizeof(FG_Resource) * 8 - 3)),
    REF_RENDERBUFFER = ((FG_Resource)4 << (sizeof(FG_Resource) * 8 - 3)),
    REF_SAMPLER      = ((FG_Resource)5 << (sizeof(FG_Resource) * 8 - 3)),
    REF_MASK         = ~((FG_Resource)0b111 << (sizeof(FG_Resource) * 8 - 3)),
  };

  struct BindRef
  {
    typedef void (*BINDRESET)(GLenum target);

    constexpr BindRef() : _target(0), _reset(nullptr) {}
    constexpr BindRef(GLenum target, BINDRESET reset) noexcept : _target(target), _reset(reset) {}
    constexpr BindRef(BindRef&& right) noexcept : _target(right._target), _reset(right._reset)
    {
      right._target = 0;
      right._reset  = nullptr;
    }
    BindRef(const BindRef&) = delete;
    void release()
    {
      _target = 0;
      _reset  = nullptr;
    }
    inline bool constexpr is_valid() const noexcept { return _target != 0; }

    constexpr BindRef& operator=(BindRef&& right) noexcept
    {
      _target       = right._target;
      _reset        = right._reset;
      right._target = 0;
      right._reset  = nullptr;
      return *this;
    }
    BindRef& operator=(const BindRef&) = delete;
    inline operator bool() const noexcept { return is_valid(); }

    GLenum _target;
    BINDRESET _reset;

    ~BindRef()
    {
      if(is_valid())
      {
        assert(_reset != nullptr);
        (*_reset)(_target);
#ifdef _DEBUG
        _reset = nullptr; // catch double destruction
#endif
      }
    }
  };

  struct Ref
  {
    constexpr Ref() noexcept : _ref(0) {}
    constexpr Ref(const Ref&) = default;
    explicit constexpr Ref(GLuint ref) noexcept : _ref(ref) {}
    inline constexpr GLuint get() noexcept { return _ref; }
    bool constexpr empty() const noexcept { return _ref == 0; }
    constexpr void swap(Ref& right) noexcept { std::swap(_ref, right._ref); }

    inline operator GLuint() const noexcept { return _ref; }
    // inline operator bool() const noexcept { return is_valid(); }
    bool constexpr operator==(const Ref& right) const noexcept { return _ref == right._ref; }
    bool constexpr operator!=(const Ref& right) const noexcept { return _ref != right._ref; }
    Ref& operator=(const Ref&) = default;

  protected:
    GLuint _ref;
  };

  typedef void (*DESTROY_FUNC)(GLuint);

  template<typename T>
  concept IContainer = requires(T v)
  {
    (*T::DESTROY)(static_cast<GLuint>(v));
    {
      v.empty()
      } -> std::same_as<bool>;
  };

  template<typename T> struct Owned : public T
  {
    constexpr Owned() noexcept : T() {}
    explicit constexpr Owned(T ref) noexcept : T(ref) {}
    explicit constexpr Owned(GLuint ref) noexcept : T(ref) {}
    explicit constexpr Owned(FG_Resource ref) noexcept : T(ref) {}
    constexpr Owned(Owned&& right) noexcept : T(right._move()) {}
    constexpr Owned(const Owned&) = delete;
    constexpr ~Owned() noexcept { reset().has_value(); }
    inline constexpr FG_Resource release() noexcept
    {
      FG_Resource ref = (*this);
      T::operator=(T());
      return ref;
    }
    GLExpected<void> constexpr reset() noexcept
    {
      if(T::empty())
      {
        (*T::DESTROY)(T::get());
        T::operator=(T());
        GL_ERROR("Deletion failure");
      }

      return {};
    }

    Owned& operator=(Owned&& right) noexcept
    {
      reset().has_value();
      T::operator=(T(right._move()));
      return *this;
    }
    Owned& operator=(const Owned&) = delete;
    const T& operator*() const noexcept { return *this; }
    T& operator*() noexcept { return *this; }
    const T* operator->() const noexcept { return this; }
    T* operator->() noexcept { return this; }
    inline operator Owned&() & noexcept { return *this; }
    inline operator const Owned&() const& noexcept { return *this; }
    inline operator Owned&&() && noexcept { return *this; }
    inline operator const Owned&&() const&& noexcept { return *this; }
    bool constexpr operator==(const Owned& right) const noexcept { return T::operator==(right); }
    bool constexpr operator!=(const Owned& right) const noexcept { return T::operator!=(right); }

    constexpr void swap(Owned& right) noexcept { T::swap(right); }

  private:
    inline constexpr GLuint _move() noexcept
    {
      GLuint ref = T::get();
      T::operator=(T());
      return ref;
    }
  };
}

#endif
