// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __DELEGATE_H__BSS__
#define __DELEGATE_H__BSS__

#include "bss_defines.h"
#include <functional>

namespace bss_util {
  // delegate class using variadic templates
#ifdef BSS_VARIADIC_TEMPLATES
  template<typename R, typename... Args>
  class BSS_COMPILER_DLLEXPORT delegate
  {
    inline delegate(std::function<R(Args...)>&& src) BSS_DELETEFUNC // Don't do delegate([&](){ return; }) or it'll go out of scope.
    typedef R RTYPE;
    typedef R(MSC_FASTCALL *GCC_FASTCALL FUNCTYPE)(void*, Args...);
  public:
    inline delegate(const delegate& copy) noexcept : _src(copy._src), _stub(copy._stub) {}
    inline delegate(void* src, R(MSC_FASTCALL *GCC_FASTCALL stub)(void*, Args...)) noexcept : _src(src), _stub(stub) {}
    inline delegate(std::function<R(Args...)>& src) noexcept : _src(&src), _stub(&stublambda) {}
    inline R operator()(Args ... args) const { return (*_stub)(_src, args...); }
    inline delegate& operator=(const delegate& right) noexcept { _src = right._src; _stub = right._stub; return *this; }
    inline bool IsEmpty() const noexcept { return _src == 0 || _stub == 0; }

    template<class T, RTYPE(MSC_FASTCALL T::*GCC_FASTCALL F)(Args...)>
    inline static delegate From(T* src) noexcept { return delegate(src, &stub<T, F>); }
    template<class T, RTYPE(MSC_FASTCALL T::*GCC_FASTCALL F)(Args...) const>
    inline static delegate From(const T* src) noexcept { return delegate(const_cast<T*>(src), &stubconst<T, F>); }
    template<RTYPE(MSC_FASTCALL *GCC_FASTCALL F)(Args...)>
    inline static delegate FromC() noexcept { return delegate(0, &stubstateless<F>); }
    template<class T, RTYPE(MSC_FASTCALL *GCC_FASTCALL F)(T*, Args...)>
    inline static delegate FromC(T* src) noexcept { return delegate((void*)src, (FUNCTYPE)&stubC<T, F>); }
    template<class T, RTYPE(MSC_FASTCALL *GCC_FASTCALL F)(const T*, Args...)>
    inline static delegate FromC(const T* src) noexcept { return delegate(const_cast<T*>(src), &stubconstC<T, F>); }

  protected:
    void* _src;
    R(MSC_FASTCALL *GCC_FASTCALL _stub)(void*, Args...);

    template <class T, RTYPE(MSC_FASTCALL T::*GCC_FASTCALL F)(Args...)>
    static R BSS_FASTCALL stub(void* src, Args... args) { return (static_cast<T*>(src)->*F)(args...); }
    template <class T, RTYPE(MSC_FASTCALL T::*GCC_FASTCALL F)(Args...) const>
    static R BSS_FASTCALL stubconst(void* src, Args... args) { return (static_cast<const T*>(src)->*F)(args...); }
    static R BSS_FASTCALL stublambda(void* src, Args... args) { return (*static_cast<std::function<R(Args...)>*>(src))(args...); }
    template <RTYPE(MSC_FASTCALL *GCC_FASTCALL F)(Args...)>
    static R BSS_FASTCALL stubstateless(void* src, Args... args) { return (*F)(args...); }
    template <class T, RTYPE(MSC_FASTCALL *GCC_FASTCALL F)(T*, Args...)>
    static R BSS_FASTCALL stubC(void* src, Args... args) { return (*F)(static_cast<T*>(src), args...); }
    template <class T, RTYPE(MSC_FASTCALL *GCC_FASTCALL F)(const T*, Args...)>
    static R BSS_FASTCALL stubconstC(void* src, Args... args) { return (*F)(static_cast<const T*>(src), args...); }
  };
#else
  template<typename R = void, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void, typename T5 = void>
  class BSS_COMPILER_DLLEXPORT delegate {};

  // delegate class done via macros because Microsoft is full of shit.
  template<typename R>
  class BSS_COMPILER_DLLEXPORT delegate<R, void, void, void, void>
  {
    inline delegate(std::function<R(void)>&& src) { assert(false); } // Don't do delegate([&](){ return; }) or it'll go out of scope.
  public:
    inline delegate(const delegate& copy) : _src(copy._src), _stub(copy._stub) {}
    inline delegate(void* src, R(MSC_FASTCALL *GCC_FASTCALL stub)(void*)) : _src(src), _stub(stub) {}
    inline delegate(std::function<R(void)>& src) : _src(&src), _stub(&stublambda) {}
    inline R operator()(void) const { return (*_stub)(_src); }
    inline delegate& operator=(const delegate& right) { _src = right._src; _stub = right._stub; return *this; }

    template<class T, R(MSC_FASTCALL T::*GCC_FASTCALL F)(void)>
    inline static delegate From(T* src) { return delegate(src, &stub<T, F>); }
    template<class T, R(MSC_FASTCALL T::*GCC_FASTCALL F)(void) const>
    inline static delegate From(const T* src) { return delegate(const_cast<T*>(src), &stubconst<T, F>); }

  protected:
    void* _src;
    R(MSC_FASTCALL *GCC_FASTCALL _stub)(void*);

    template <class T, R(MSC_FASTCALL T::*GCC_FASTCALL F)(void)>
    static R BSS_FASTCALL stub(void* src) { return (static_cast<T*>(src)->*F)(); }
    template <class T, R(MSC_FASTCALL T::*GCC_FASTCALL F)(void) const>
    static R BSS_FASTCALL stubconst(void* src) { return (static_cast<const T*>(src)->*F)(); }
    static R BSS_FASTCALL stublambda(void* src) { return (*static_cast<std::function<R(void)>*>(src))(); }
  };

#define BUILD_DELEGATE(T1,T2,T3,T4,TLIST,ARGLIST,ARGS,...) template<typename R,__VA_ARGS__> \
  class BSS_COMPILER_DLLEXPORT delegate<R,T1,T2,T3,T4> \
  { \
  public: \
    inline delegate(void* src, R (MSC_FASTCALL *GCC_FASTCALL stub)(void*,TLIST)) : _src(src), _stub(stub) {} \
    inline delegate(std::function<R(TLIST)>& src):_src(&src), _stub(&stublambda) {} \
    R operator()(ARGLIST) const { return (*_stub)(_src,ARGS); } \
    template<class T, R (MSC_FASTCALL T::*GCC_FASTCALL F)(TLIST)> \
    inline static delegate From(T* src) { return delegate(src, &stub<T, F>); } \
    template<class T, R (MSC_FASTCALL T::*GCC_FASTCALL F)(TLIST) const> \
    inline static delegate From(const T* src) { return delegate(const_cast<T*>(src), &stubconst<T, F>); } \
  protected: \
    void* _src; \
    R (MSC_FASTCALL *GCC_FASTCALL _stub)(void*,TLIST); \
    template <class T, R (MSC_FASTCALL T::*GCC_FASTCALL F)(TLIST)> \
    static R BSS_FASTCALL stub(void* src, ARGLIST) { return (static_cast<T*>(src)->*F)(ARGS); } \
    template <class T, R (MSC_FASTCALL T::*GCC_FASTCALL F)(TLIST) const> \
    static R BSS_FASTCALL stubconst(void* src, ARGLIST) { return (static_cast<const T*>(src)->*F)(ARGS); } \
    static R BSS_FASTCALL stublambda(void* src, ARGLIST) { return (*static_cast<std::function<R(TLIST)>*>(src))(ARGS); } \
  }

  BUILD_DELEGATE(T1, void, void, void, CONCAT(T1), CONCAT(T1 t1), CONCAT(t1), typename T1);
  BUILD_DELEGATE(T1, T2, void, void, CONCAT(T1, T2), CONCAT(T1 t1, T2 t2), CONCAT(t1, t2), typename T1, typename T2);
  BUILD_DELEGATE(T1, T2, T3, void, CONCAT(T1, T2, T3), CONCAT(T1 t1, T2 t2, T3 t3), CONCAT(t1, t2, t3), typename T1, typename T2, typename T3);
  BUILD_DELEGATE(T1, T2, T3, T4, CONCAT(T1, T2, T3, T4), CONCAT(T1 t1, T2 t2, T3 t3, T4 t4), CONCAT(t1, t2, t3, t4), typename T1, typename T2, typename T3, typename T4);
#endif
}

#endif
