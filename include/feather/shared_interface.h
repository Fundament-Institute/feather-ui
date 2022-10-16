/* graphics_interface.h - Shared C types for feather interfaces
Copyright (c)2022 Fundament Software

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SHARED_INTERFACE_H
#define SHARED_INTERFACE_H

#include <stdint.h>  // for integers
#include <stdbool.h> // for bool

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FG_Vec3i__
{
  int x;
  int y;
  int z;
} FG_Vec3i;

typedef struct FG_Vec3__
{
  float x;
  float y;
  float z;
} FG_Vec3;

typedef struct FG_Vec2i__
{
  int x;
  int y;
} FG_Vec2i;

typedef struct FG_Vec2__
{
  float x;
  float y;
} FG_Vec2;

typedef union FG_Rect__
{
  struct
  {
    float left;
    float top;
    float right;
    float bottom;
  };

  float ltrb[4];
} FG_Rect;

typedef union FG_Color8__
{
  uint32_t v;
  uint8_t colors[4];
  struct
  {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
  };
} FG_Color8;

typedef union FG_Color16__
{
  uint64_t v;
  uint16_t colors[4];
  struct
  {
    uint16_t b;
    uint16_t g;
    uint16_t r;
    uint16_t a;
  };
} FG_Color16;

enum FG_Level
{
  FG_Level_Fatal   = 0,
  FG_Level_Error   = 1,
  FG_Level_Warning = 2,
  FG_Level_Notice  = 3,
  FG_Level_Debug   = 4,
};

enum FG_LogType
{
  FG_LogType_Boolean,
  FG_LogType_I32,
  FG_LogType_U32,
  FG_LogType_I64,
  FG_LogType_U64,
  FG_LogType_F32,
  FG_LogType_F64,
  FG_LogType_String,
  FG_LogType_OwnedString,
};

typedef struct FG_LogValue__
{
  enum FG_LogType type;
  union
  {
    bool bit;
    int i32;
    unsigned int u32;
    long long i64;
    unsigned long long u64;
    float f32;
    double f64;
    const char* string;
    char* owned;
  };
} FG_LogValue;

typedef void FG_Context;
typedef void (*FG_Log)(void*, enum FG_Level, const char*, int, const char*, const FG_LogValue*, int, void (*)(char*));

#ifdef __cplusplus
}

#include <type_traits>
#include <memory>
#include <string>

template<class T> inline constexpr FG_LogValue fgLogValue(const T& v)
{
  using U = std::remove_cvref_t<T>;
  // We do not use is_same for anything other than bool, because types like "unsigned long long" and "uint64_t" might
  // not actually be the same type depending on the compiler
  if constexpr(std::is_same_v<U, bool>)
    return FG_LogValue{ .type = FG_LogType_Boolean, .bit = v };
  else if constexpr(std::is_same_v<T, const char*>)
    return FG_LogValue{ .type = FG_LogType_String, .string = v };
  else if constexpr(std::is_same_v<U, std::string>)
  {
    char* str = malloc(v.size() + 1);
    strcpy(str, v.data());
    return FG_LogValue{ .type = FG_LogType_String, .owned = str };
  }
  else if constexpr(std::is_same_v<U, std::unique_ptr<char>>)
    return FG_LogValue{ .type = FG_LogType_String, .owned = v.release() };
  else if constexpr(std::is_floating_point_v<U> && sizeof(U) == sizeof(float))
    return FG_LogValue{ .type = FG_LogType_F32, .f32 = v };
  else if constexpr(std::is_floating_point_v<U> && sizeof(U) == sizeof(double))
    return FG_LogValue{ .type = FG_LogType_F64, .f64 = v };
  else if constexpr(std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == sizeof(int32_t))
    return FG_LogValue{ .type = FG_LogType_I32, .i32 = v };
  else if constexpr(std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == sizeof(int32_t))
    return FG_LogValue{ .type = FG_LogType_U32, .u32 = v };
  else if constexpr(std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == sizeof(int64_t))
    return FG_LogValue{ .type = FG_LogType_I64, .i64 = v };
  else if constexpr(std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == sizeof(int64_t))
    return FG_LogValue{ .type = FG_LogType_U64, .u64 = v };
  else
    static_assert(std::is_same_v<U, bool>, "Invalid type for FG_LogValue");
}

template<void (*FREE)(char*), typename T, std::size_t... I>
inline void fgLogImpl(FG_Log log, void* ctx, FG_Level level, const char* file, int line, const char* msg, const T& args, std::index_sequence<I...>)
{
  FG_LogValue values[sizeof...(I)] = { fgLogValue(std::get<I>(args))... };

  log(ctx, level, file, line, msg, values, sizeof...(I), FREE);
}
#endif

#endif
