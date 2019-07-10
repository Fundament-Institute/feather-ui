// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__EXAMPLE_FIELD_H
#define FG__EXAMPLE_FIELD_H

#include "feather/data.h"
#include <unordered_map>
#include <string>
#include <initializer_list>
#include <functional>

// We have to build our own custom variant structure because there's no other way to recursively reference ourselves
struct Field
{
  Field() : type(FG_DATA_NONE), i(0), name(0) {}
  template<class U>
  Field(U value, const char* n) : type(type), name(n)
  {
    if constexpr(std::is_same<U, float>::value)
    {
      d = value;
      type = FG_DATA_FLOAT;
    }
    else if constexpr(std::is_same<U, double>::value)
    {
      d = value;
      type = FG_DATA_DOUBLE;
    }
    else if constexpr(std::is_same<U, int>::value)
    {
      i = value;
      type = FG_DATA_INT32;
    }
    else if constexpr(std::is_integral<U>::value)
    {
      i = value;
      type = FG_DATA_INT64;
    }
    else if constexpr(std::is_same<U, const char*>::value || std::is_same<std::remove_cvref<U>, std::string>::value)
    {
      new (&s) std::string(value);
      type = FG_DATA_ARRAY | FG_DATA_CHAR;
    }
    else
      static_assert(false, "Invalid type");
  }
  Field(std::function<void()> lambda) : f(lambda), type(FG_DATA_FUNCTION), name(0) {}
  Field(FG_DATA_TYPES type) : type(type), i(0), name(0)
  {
    switch(type)
    {
    case FG_DATA_ARRAY | FG_DATA_CHAR:
    case FG_DATA_ARRAY | FG_DATA_BYTE:
      new (&s) std::string();
      break;
    case FG_DATA_OBJECT:
      new (&o) std::unordered_map<std::string, Field>();
      break;
    case FG_DATA_FUNCTION:
      new (&f) std::function<void()>();
      break;
    default:
      if(type & FG_DATA_ARRAY)
        new (&a) std::vector<Field>();
      break;
    }
  }
  Field(std::initializer_list<std::pair<const std::string, Field>> list) : type(FG_DATA_OBJECT), name("object")
  {
    new (&o) std::unordered_map<std::string, Field>(list);
  }
  Field(const Field& copy) : type(copy.type), name(copy.name)
  {
    switch(type)
    {
    case FG_DATA_ARRAY | FG_DATA_CHAR:
    case FG_DATA_ARRAY | FG_DATA_BYTE:
      new (&s) std::string(copy.s);
      break;
    case FG_DATA_OBJECT:
      new (&o) std::unordered_map<std::string, Field>(copy.o);
      break;
    case FG_DATA_FUNCTION:
      new (&f) std::function<void()>(copy.f);
      break;
    default:
      if(type & FG_DATA_ARRAY)
        new (&a) std::vector<Field>(copy.a);
      else
        i = copy.i;
      break;
    }
  }
  ~Field()
  {
    switch(type)
    {
    case FG_DATA_ARRAY | FG_DATA_CHAR:
    case FG_DATA_ARRAY | FG_DATA_BYTE:
      s.~basic_string();
      break;
    case FG_DATA_OBJECT:
      o.~unordered_map();
      break;
    case FG_DATA_FUNCTION:
      f.~function();
      break;
    default:
      if(type & FG_DATA_ARRAY)
        a.~vector();
      break;
    }
    type = FG_DATA_NONE;
  }
  Field& operator=(const Field& r)
  {
    this->~Field();
    new (this) Field(r);
    return *this;
  }

  int type;
  const char* name;
  union
  {
    std::string s;
    std::function<void()> f;
    double d;
    int64_t i;
    void* p;
    std::unordered_map<std::string, Field> o;
    std::vector<Field> a;
  };
 
  fgDataField data() const
  {
    fgDataField field = { "None", name, (void*)this, type };
    switch(type)
    {
    case FG_DATA_FLOAT: field.type = "f32"; field.data.f = d; break;
    case FG_DATA_DOUBLE: field.type = "f64"; field.data.d = d; break;
    case FG_DATA_INT32: field.type = "i32"; field.data.i = i; break;
    case FG_DATA_INT64: field.type = "i64"; field.data.l = i; break;
    case FG_DATA_COMPLEX: field.type = "complex"; field.data.ptr = p; break;
    case FG_DATA_FUNCTION: field.type = "fn"; field.data.func = &EvalLambda; break;
    case FG_DATA_CHAR: field.type = "char"; field.data.c = i; break;
    case FG_DATA_BYTE: field.type = "byte"; field.data.b = i; break;
    case FG_DATA_OBJECT: field.type = "object"; break;
    case FG_DATA_ARRAY | FG_DATA_FLOAT: field.type = "[f32]"; field.data.sz = a.size(); break;
    case FG_DATA_ARRAY | FG_DATA_DOUBLE: field.type = "[f64]"; field.data.sz = a.size(); break;
    case FG_DATA_ARRAY | FG_DATA_INT32: field.type = "[i32]"; field.data.sz = a.size(); break;
    case FG_DATA_ARRAY | FG_DATA_INT64: field.type = "[i64]"; field.data.sz = a.size(); break;
    case FG_DATA_ARRAY | FG_DATA_COMPLEX: field.type = "[complex]"; field.data.sz = a.size(); break;
    case FG_DATA_ARRAY | FG_DATA_FUNCTION: field.type = "[fn]"; field.data.sz = a.size(); break;
    case FG_DATA_ARRAY | FG_DATA_CHAR: field.type = "[char]"; field.data.sz = s.size(); break;
    case FG_DATA_ARRAY | FG_DATA_BYTE: field.type = "[byte]"; field.data.sz = s.size(); break;
    }
    return field;
  }

  static void EvalLambda(void* obj)
  {
    auto p = reinterpret_cast<Field*>(obj);
    if(p->type == FG_DATA_FUNCTION && p->f)
      p->f();
  }

  static fgDataField GetField(void* obj, const char* field)
  {
    auto p = reinterpret_cast<Field*>(obj);
    if(!field)
      return p->data();
    if(p->type != FG_DATA_OBJECT)
      return fgDataField{};

    if(auto i = p->o.find(field); i != p->o.end())
      return i->second.data();
    return fgDataField{};
  }

  static fgDataField GetIndex(void* obj, unsigned int index)
  {
    auto p = reinterpret_cast<Field*>(obj);
    if(!(p->type & FG_DATA_ARRAY))
      return fgDataField{};
    switch(p->type & FG_DATA_ELEMENT_MASK)
    {
    case FG_DATA_CHAR:
    case FG_DATA_BYTE:
      if(index == ~0)
      {
        fgDataField v = p->data();
        v.primitive = FG_DATA_ARRAY;
        v.data.ptr = p->s.data();
        return v;
      }
      if(index >= p->s.size())
        return fgDataField{};
      else
      {
        fgDataField v = fgDataField{ "char", 0, 0, FG_DATA_TYPES(p->type & FG_DATA_ELEMENT_MASK) };
        v.data.c = p->s[index];
        return v;
      }
    default:
      break;
    }

    if(index == ~0)
    {
      fgDataField v = p->data();
      v.primitive = FG_DATA_ARRAY;
      v.data.ptr = p->a.data();
      return v;
    }
    if(index >= p->a.size())
      return fgDataField{};
    return p->a[index].data();
  }

  static bool SetData(void* obj, enum FG_DATA_TYPES type, union fgDataUnion data)
  {
    auto p = reinterpret_cast<Field*>(obj);
    switch(type)
    {
    default:
      return false;
    case FG_DATA_FLOAT: p->d = data.f; break;
    case FG_DATA_DOUBLE: p->d = data.d; break;
    case FG_DATA_INT32: p->i = data.i; break;
    case FG_DATA_INT64: p->i = data.l; break;
    case FG_DATA_FUNCTION: p->p = data.func; break;
    case FG_DATA_CHAR: p->i = data.c; break;
    case FG_DATA_BYTE: p->i = data.b; break;
    case FG_DATA_COMPLEX: p->p = data.ptr; break;
    }
    p->type = type;
    return true;
  }

  static bool SetRange(void* obj, enum FG_DATA_TYPES element, void* data, unsigned long long length, unsigned int offset, unsigned int count)
  {
    auto p = reinterpret_cast<Field*>(obj);
    if(!(p->type & FG_DATA_ARRAY))
      return false;
    switch(p->type & FG_DATA_ELEMENT_MASK)
    {
    case FG_DATA_BYTE:
    case FG_DATA_CHAR:
      if(offset >= p->s.size())
        return false;
      if(!length)
        p->s.erase(offset, count);
      else if(!count)
        p->s.insert(offset, (const char*)data, length);
      else
        p->s.replace(offset, count, (const char*)data, length);
      break;
    default:
      if(offset >= p->a.size())
        return false;
      if(!length)
        p->a.erase(p->a.begin() + offset, p->a.begin() + offset + count);
      else
        return false;
    }
    return true;
  }
}; 

#endif
