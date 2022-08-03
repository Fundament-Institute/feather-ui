// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__REF_H
#define GL__REF_H

#include "GLError.hpp"

namespace GL {
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

  // template<typename T>
  // concept IRef = std::default_initializable<T> && std::convertible_to<T, GLuint> && std::constructible_from<GLuint>;

  template<bool (*IS)(GLuint)> struct Ref
  {
    constexpr Ref() noexcept : _ref(0) {}
    constexpr Ref(const Ref&) = default;
    explicit constexpr Ref(GLuint ref) noexcept : _ref(ref) {}
    inline constexpr GLuint get() noexcept { return _ref; }
    bool constexpr is_valid() const noexcept { return IS(_ref) == GL_TRUE; }
    constexpr void swap(Ref& right) noexcept { std::swap(_ref, right._ref); }

    inline operator GLuint() const noexcept { return _ref; }
    inline operator bool() const noexcept { return is_valid(); }
    bool constexpr operator==(const Ref& right) const noexcept { return _ref == right._ref; }
    bool constexpr operator!=(const Ref& right) const noexcept { return _ref != right._ref; }
    Ref& operator=(const Ref&) = default;

  protected:
    GLuint _ref;
  };

  typedef void (*DESTROY_FUNC)(GLuint);

  /* template<typename T>
  concept IContainer = IRef<T> && requires(T v)
  {
    (*T::DESTROY)(static_cast<GLuint>(v));
    { v.is_valid()} -> std::same_as<bool>;
  };*/

  template<typename T> struct Owned : public T
  {
    constexpr Owned() noexcept : T() {}
    explicit constexpr Owned(T ref) noexcept : T(ref) {}
    explicit constexpr Owned(GLuint ref) noexcept : T(ref) {}
    explicit constexpr Owned(void* ref) noexcept : T(static_cast<GLuint>(reinterpret_cast<std::size_t>(ref))) {}
    constexpr Owned(Owned&& right) noexcept : T(right.release()) {}
    constexpr Owned(const Owned&) = delete;
    constexpr ~Owned() noexcept { reset().has_value(); }
    inline constexpr GLuint release() noexcept
    {
      GLuint ref = T::get();
      T::operator=(T());
      return ref;
    }
    GLExpected<void> constexpr reset() noexcept
    {
      if(T::is_valid())
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
      T::operator=(T(right.release()));
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
  };
}

#endif
