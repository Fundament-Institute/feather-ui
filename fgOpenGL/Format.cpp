// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "Format.hpp"
#include <cassert>
#include <cmath>

using namespace GL;

// https://github.com/richardmg/dachshund/blob/7f5404b89aab84ae77995725d2b408109f514479/src/3rdparty/angle/src/libANGLE/es3_format_type_combinations.json

Format Format::Map(GLint internalformat) noexcept
{
  // The standard allows multiple valid type combinations for certain formats. We pick the most specific
  // type possible. The alternatives are left commented out in case we need to implement them as fallbacks.
  // Some formats are commented out because we don't have the extensions to support them enabled.
  switch(internalformat)
  {
  case GL_RGBA8: return Format{ GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE };
  // case GL_RGB5_A1: return Format{ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_BYTE };
  // case GL_RGBA4: return Format{ GL_RGBA4, GL_RGBA, GL_UNSIGNED_BYTE };
  case GL_SRGB8_ALPHA8: return Format{ GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE };
  case GL_RGBA8_SNORM: return Format{ GL_RGBA8_SNORM, GL_RGBA, GL_BYTE };
  case GL_RGBA4: return Format{ GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4 };
  case GL_RGB10_A2: return Format{ GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  // case GL_RGB5_A1: return Format{ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  case GL_RGB5_A1: return Format{ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1 };
  case GL_RGBA16F: return Format{ GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT };
  // case GL_RGBA16F: return Format{ GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT_OES };
  case GL_RGBA32F: return Format{ GL_RGBA32F, GL_RGBA, GL_FLOAT };
  // case GL_RGBA16F: return Format{ GL_RGBA16F, GL_RGBA, GL_FLOAT };
  case GL_RGBA8UI: return Format{ GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE };
  case GL_RGBA8I: return Format{ GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE };
  case GL_RGBA16UI: return Format{ GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT };
  case GL_RGBA16I: return Format{ GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT };
  case GL_RGBA32UI: return Format{ GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT };
  case GL_RGBA32I: return Format{ GL_RGBA32I, GL_RGBA_INTEGER, GL_INT };
  // case GL_RGB10_A2UI: return Format{ GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV };
  case GL_RGB8: return Format{ GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE };
  // case GL_RGB565: return Format{ GL_RGB565, GL_RGB, GL_UNSIGNED_BYTE };
  case GL_SRGB8: return Format{ GL_SRGB8, GL_RGB, GL_UNSIGNED_BYTE };
  case GL_RGB8_SNORM: return Format{ GL_RGB8_SNORM, GL_RGB, GL_BYTE };
  // case GL_RGB565: return Format{ GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5 };
  case GL_R11F_G11F_B10F: return Format{ GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV };
  case GL_RGB9_E5: return Format{ GL_RGB9_E5, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV };
  case GL_RGB16F: return Format{ GL_RGB16F, GL_RGB, GL_HALF_FLOAT };
  // case GL_RGB16F: return Format{ GL_RGB16F, GL_RGB, GL_HALF_FLOAT_OES };
  // case GL_R11F_G11F_B10F: return Format{ GL_R11F_G11F_B10F, GL_RGB, GL_HALF_FLOAT };
  // case GL_R11F_G11F_B10F: return Format{ GL_R11F_G11F_B10F, GL_RGB, GL_HALF_FLOAT_OES };
  // case GL_RGB9_E5: return Format{ GL_RGB9_E5, GL_RGB, GL_HALF_FLOAT };
  // case GL_RGB9_E5: return Format{ GL_RGB9_E5, GL_RGB, GL_HALF_FLOAT_OES };
  case GL_RGB32F: return Format{ GL_RGB32F, GL_RGB, GL_FLOAT };
  // case GL_RGB16F: return Format{ GL_RGB16F, GL_RGB, GL_FLOAT };
  // case GL_R11F_G11F_B10F: return Format{ GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT };
  // case GL_RGB9_E5: return Format{ GL_RGB9_E5, GL_RGB, GL_FLOAT };
  case GL_RGB8UI: return Format{ GL_RGB8UI, GL_RGB_INTEGER, GL_UNSIGNED_BYTE };
  case GL_RGB8I: return Format{ GL_RGB8I, GL_RGB_INTEGER, GL_BYTE };
  case GL_RGB16UI: return Format{ GL_RGB16UI, GL_RGB_INTEGER, GL_UNSIGNED_SHORT };
  case GL_RGB16I: return Format{ GL_RGB16I, GL_RGB_INTEGER, GL_SHORT };
  case GL_RGB32UI: return Format{ GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT };
  case GL_RGB32I: return Format{ GL_RGB32I, GL_RGB_INTEGER, GL_INT };
  case GL_RG8: return Format{ GL_RG8, GL_RG, GL_UNSIGNED_BYTE };
  case GL_RG8_SNORM: return Format{ GL_RG8_SNORM, GL_RG, GL_BYTE };
  case GL_RG16F: return Format{ GL_RG16F, GL_RG, GL_HALF_FLOAT };
  // case GL_RG16F: return Format{ GL_RG16F, GL_RG, GL_HALF_FLOAT_OES };
  case GL_RG32F: return Format{ GL_RG32F, GL_RG, GL_FLOAT };
  // case GL_RG16F: return Format{ GL_RG16F, GL_RG, GL_FLOAT };
  case GL_RG8UI: return Format{ GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE };
  case GL_RG8I: return Format{ GL_RG8I, GL_RG_INTEGER, GL_BYTE };
  case GL_RG16UI: return Format{ GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT };
  case GL_RG16I: return Format{ GL_RG16I, GL_RG_INTEGER, GL_SHORT };
  case GL_RG32UI: return Format{ GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT };
  case GL_RG32I: return Format{ GL_RG32I, GL_RG_INTEGER, GL_INT };
  case GL_R8: return Format{ GL_R8, GL_RED, GL_UNSIGNED_BYTE };
  case GL_R8_SNORM: return Format{ GL_R8_SNORM, GL_RED, GL_BYTE };
  case GL_R16F: return Format{ GL_R16F, GL_RED, GL_HALF_FLOAT };
  // case GL_R16F: return Format{ GL_R16F, GL_RED, GL_HALF_FLOAT_OES };
  case GL_R32F: return Format{ GL_R32F, GL_RED, GL_FLOAT };
  // case GL_R16F: return Format{ GL_R16F, GL_RED, GL_FLOAT };
  case GL_R8UI: return Format{ GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE };
  case GL_R8I: return Format{ GL_R8I, GL_RED_INTEGER, GL_BYTE };
  case GL_R16UI: return Format{ GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT };
  case GL_R16I: return Format{ GL_R16I, GL_RED_INTEGER, GL_SHORT };
  case GL_R32UI: return Format{ GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT };
  case GL_R32I: return Format{ GL_R32I, GL_RED_INTEGER, GL_INT };
  case GL_RGBA: return Format{ GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE };
  // case GL_RGBA: return Format{ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4 };
  // case GL_RGBA: return Format{ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1 };
  case GL_RGB: return Format{ GL_RGB, GL_RGB, GL_UNSIGNED_BYTE };
  // case GL_RGB: return Format{ GL_RGB, GL_RGB, GL_UNSIGNED_SHORT_5_6_5 };
  case GL_LUMINANCE_ALPHA: return Format{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE };
  case GL_LUMINANCE: return Format{ GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE };
  case GL_ALPHA: return Format{ GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE };
  // case GL_SRGB_ALPHA_EXT: return Format{ GL_SRGB_ALPHA_EXT, GL_SRGB_ALPHA_EXT, GL_UNSIGNED_BYTE };
  // case GL_SRGB_EXT: return Format{ GL_SRGB_EXT, GL_SRGB_EXT, GL_UNSIGNED_BYTE };
  case GL_RG: return Format{ GL_RG, GL_RG, GL_UNSIGNED_BYTE };
  // case GL_RG: return Format{ GL_RG, GL_RG, GL_FLOAT };
  // case GL_RG: return Format{ GL_RG, GL_RG, GL_HALF_FLOAT };
  // case GL_RG: return Format{ GL_RG, GL_RG, GL_HALF_FLOAT_OES };
  case GL_RED: return Format{ GL_RED, GL_RED, GL_UNSIGNED_BYTE };
  // case GL_RED: return Format{ GL_RED, GL_RED, GL_FLOAT };
  // case GL_RED: return Format{ GL_RED, GL_RED, GL_HALF_FLOAT };
  // case GL_RED: return Format{ GL_RED, GL_RED, GL_HALF_FLOAT_OES };
  case GL_DEPTH_STENCIL: return Format{ GL_DEPTH_STENCIL, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 };
  case GL_DEPTH_COMPONENT16: return Format{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT };
  case GL_DEPTH_COMPONENT24: return Format{ GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
  // case GL_DEPTH_COMPONENT16: return Format{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
  case GL_DEPTH_COMPONENT32F: return Format{ GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT };
  case GL_DEPTH24_STENCIL8: return Format{ GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 };
  case GL_DEPTH32F_STENCIL8: return Format{ GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV };
  // case GL_DEPTH_COMPONENT32_OES: return Format{ GL_DEPTH_COMPONENT32_OES, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT_24_8 };
  // case GL_DEPTH_COMPONENT: return Format{ GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT };
  case GL_DEPTH_COMPONENT:
    return Format{ GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
    // case GL_SRGB8_ALPHA8_EXT: return Format{ GL_SRGB8_ALPHA8_EXT, GL_SRGB_ALPHA_EXT, GL_UNSIGNED_BYTE };
    /* case GL_SRGB8:
      return Format{ GL_SRGB8, GL_SRGB_EXT, GL_UNSIGNED_BYTE };
    case GL_RGBA: return Format{ GL_RGBA, GL_RGBA, GL_FLOAT };
    case GL_RGB: return Format{ GL_RGB, GL_RGB, GL_FLOAT };
    case GL_LUMINANCE_ALPHA: return Format{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_FLOAT };
    case GL_LUMINANCE: return Format{ GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT };
    case GL_ALPHA: return Format{ GL_ALPHA, GL_ALPHA, GL_FLOAT };
    case GL_RGBA: return Format{ GL_RGBA, GL_RGBA, GL_HALF_FLOAT_OES };
    case GL_RGB: return Format{ GL_RGB, GL_RGB, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE_ALPHA: return Format{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT };
    case GL_LUMINANCE_ALPHA: return Format{ GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE: return Format{ GL_LUMINANCE, GL_LUMINANCE, GL_HALF_FLOAT };
    case GL_LUMINANCE: return Format{ GL_LUMINANCE, GL_LUMINANCE, GL_HALF_FLOAT_OES };
    case GL_ALPHA: return Format{ GL_ALPHA, GL_ALPHA, GL_HALF_FLOAT };
    case GL_ALPHA: return Format{ GL_ALPHA, GL_ALPHA, GL_HALF_FLOAT_OES };
    case GL_BGRA_EXT: return Format{ GL_BGRA_EXT, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_ALPHA8_EXT: return Format{ GL_ALPHA8_EXT, GL_ALPHA, GL_UNSIGNED_BYTE };
    case GL_LUMINANCE8_EXT: return Format{ GL_LUMINANCE8_EXT, GL_LUMINANCE, GL_UNSIGNED_BYTE };
    case GL_LUMINANCE8_ALPHA8_EXT: return Format{ GL_LUMINANCE8_ALPHA8_EXT, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE };
    case GL_ALPHA32F_EXT: return Format{ GL_ALPHA32F_EXT, GL_ALPHA, GL_FLOAT };
    case GL_LUMINANCE32F_EXT: return Format{ GL_LUMINANCE32F_EXT, GL_LUMINANCE, GL_FLOAT };
    case GL_LUMINANCE_ALPHA32F_EXT: return Format{ GL_LUMINANCE_ALPHA32F_EXT, GL_LUMINANCE_ALPHA, GL_FLOAT };
    case GL_ALPHA16F_EXT: return Format{ GL_ALPHA16F_EXT, GL_ALPHA, GL_HALF_FLOAT };
    case GL_ALPHA16F_EXT: return Format{ GL_ALPHA16F_EXT, GL_ALPHA, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE16F_EXT: return Format{ GL_LUMINANCE16F_EXT, GL_LUMINANCE, GL_HALF_FLOAT };
    case GL_LUMINANCE16F_EXT: return Format{ GL_LUMINANCE16F_EXT, GL_LUMINANCE, GL_HALF_FLOAT_OES };
    case GL_LUMINANCE_ALPHA16F_EXT: return Format{ GL_LUMINANCE_ALPHA16F_EXT, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT };
    case GL_LUMINANCE_ALPHA16F_EXT: return Format{ GL_LUMINANCE_ALPHA16F_EXT, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT_OES };
    case GL_BGRA8_EXT: return Format{ GL_BGRA8_EXT, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_BGRA4_ANGLEX: return Format{ GL_BGRA4_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT };
    case GL_BGRA4_ANGLEX: return Format{ GL_BGRA4_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_BGR5_A1_ANGLEX: return Format{ GL_BGR5_A1_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT };
    case GL_BGR5_A1_ANGLEX: return Format{ GL_BGR5_A1_ANGLEX, GL_BGRA_EXT, GL_UNSIGNED_BYTE };
    case GL_R16_EXT: return Format{ GL_R16_EXT, GL_RED, GL_UNSIGNED_SHORT };
    case GL_RG16_EXT: return Format{ GL_RG16_EXT, GL_RG, GL_UNSIGNED_SHORT };
    case GL_RGB16_EXT: return Format{ GL_RGB16_EXT, GL_RGB, GL_UNSIGNED_SHORT };
    case GL_RGBA16_EXT: return Format{ GL_RGBA16_EXT, GL_RGBA, GL_UNSIGNED_SHORT };
    case GL_R16_SNorm_EXT: return Format{ GL_R16_SNorm_EXT, GL_RED, GL_SHORT };
    case GL_RG16_SNorm_EXT: return Format{ GL_RG16_SNorm_EXT, GL_RG, GL_SHORT };
    case GL_RGB16_SNorm_EXT: return Format{ GL_RGB16_SNorm_EXT, GL_RGB, GL_SHORT };
    case GL_RGBA16_SNorm_EXT: return Format{ GL_RGBA16_SNorm_EXT, GL_RGBA, GL_SHORT };
    */
  }
  return Format{ 0, 0, 0 };
}

Format Format::Create(unsigned char format, bool sRGB) noexcept
{
  switch(format)
  {
  case FG_PixelFormat_Unknown: return Format{ 0, 0, 0 };
  case FG_PixelFormat_R32G32B32A32_Typeless: return Format{ GL_RGBA, GL_RGBA, GL_FLOAT };
  case FG_PixelFormat_R32G32B32A32_Float: return Format{ GL_RGBA32F, GL_RGBA, GL_FLOAT };
  case FG_PixelFormat_R32G32B32A32_UInt: return Format{ GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32B32A32_Int: return Format{ GL_RGBA32I, GL_RGBA_INTEGER, GL_INT };
  case FG_PixelFormat_R32G32B32_Typeless: return Format{ GL_RGB, GL_RGB, GL_FLOAT };
  case FG_PixelFormat_R32G32B32_Float: return Format{ GL_RGB32F, GL_RGB, GL_FLOAT };
  case FG_PixelFormat_R32G32B32_UInt: return Format{ GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32B32_Int: return Format{ GL_RGB32I, GL_RGB_INTEGER, GL_INT };
  case FG_PixelFormat_R16G16B16A16_Typeless: return Format{ GL_RGBA16, GL_RGBA, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16A16_Float: return Format{ GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16A16_UNorm: return Format{ GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16A16_UInt: return Format{ GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16A16_SNorm: return Format{ GL_RGBA16_SNORM, GL_RGBA, GL_SHORT };
  case FG_PixelFormat_R16G16B16A16_Int: return Format{ GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT };
  case FG_PixelFormat_R16G16B16_Typeless: return Format{ GL_RGB16, GL_RGB, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16_Float: return Format{ GL_RGB16F, GL_RGB, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16B16_UNorm: return Format{ GL_RGB16, GL_RGB, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16_UInt: return Format{ GL_RGB16UI, GL_RGB_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16B16_SNorm: return Format{ GL_RGB16_SNORM, GL_RGBA, GL_SHORT };
  case FG_PixelFormat_R16G16B16_Int: return Format{ GL_RGB16I, GL_RGB_INTEGER, GL_SHORT };
  case FG_PixelFormat_R32G32_Typeless: return Format{ GL_RG, GL_RG, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32_Float: return Format{ GL_RG32F, GL_RG, GL_FLOAT };
  case FG_PixelFormat_R32G32_UInt: return Format{ GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32G32_Int: return Format{ GL_RGB32I, GL_RGB_INTEGER, GL_INT };
  case FG_PixelFormat_R32G8X24_Typeless: return Format{ 0, 0, 0 };
  case FG_PixelFormat_D32_Float_S8X24_UInt:
    return Format{ GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV };
  case FG_PixelFormat_R32_Float_X8X24_Typeless: return Format{ 0, 0, 0 };
  case FG_PixelFormat_X32_Typeless_G8X24_UInt: return Format{ 0, 0, 0 };
  case FG_PixelFormat_R10G10B10A2_Typeless: return Format{ GL_RGBA, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  case FG_PixelFormat_R10G10B10A2_UNorm: return Format{ GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV };
  // case FG_PixelFormat_R10G10B10A2_UInt: return Format{ GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV };
  case FG_PixelFormat_R11G11B10_Float: return Format{ GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV };
  case FG_PixelFormat_R8G8B8A8_Typeless: return Format{ sRGB ? GL_SRGB_ALPHA : GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8A8_UNorm: return Format{ sRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8A8_UInt: return Format{ GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8A8_SNorm: return Format{ GL_RGBA8_SNORM, GL_RGBA, GL_BYTE };
  case FG_PixelFormat_R8G8B8A8_Int: return Format{ GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE };
  case FG_PixelFormat_R8G8B8X8_Typeless: return Format{ sRGB ? GL_SRGB : GL_RGB, GL_RGB, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8X8_UNorm: return Format{ sRGB ? GL_SRGB8 : GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8X8_SNorm: return Format{ GL_RGB8_SNORM, GL_RGB, GL_BYTE };
  case FG_PixelFormat_R8G8B8X8_UInt: return Format{ GL_RGB8UI, GL_RGB_INTEGER, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8B8X8_Int: return Format{ GL_RGB8I, GL_RGB_INTEGER, GL_BYTE };
  case FG_PixelFormat_R16G16_Typeless: return Format{ GL_RG16, GL_RG, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16_Float: return Format{ GL_RG16F, GL_RG, GL_HALF_FLOAT };
  case FG_PixelFormat_R16G16_UNorm: return Format{ GL_RG16, GL_RG, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16_UInt: return Format{ GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16G16_SNorm: return Format{ GL_RG16_SNORM, GL_RG, GL_SHORT };
  case FG_PixelFormat_R16G16_Int: return Format{ GL_RG16I, GL_RG_INTEGER, GL_SHORT };
  case FG_PixelFormat_R32_Typeless: return Format{ GL_RED, GL_RED, GL_FLOAT };
  case FG_PixelFormat_R32_Float: return Format{ GL_R32F, GL_RED, GL_FLOAT };
  case FG_PixelFormat_R32_UInt: return Format{ GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT };
  case FG_PixelFormat_R32_Int: return Format{ GL_R32I, GL_RED_INTEGER, GL_INT };
  case FG_PixelFormat_D32_Float: return Format{ GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, GL_FLOAT };
  case FG_PixelFormat_D24_UNorm_X8_Typeless: return Format{ GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT };
  case FG_PixelFormat_R24_UNorm_X8_Typeless: return Format{ 0, 0, 0 };
  case FG_PixelFormat_D16_UNorm: return Format{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R24G8_Typeless: return Format{ GL_RG, GL_RG, GL_UNSIGNED_INT_24_8 };
  case FG_PixelFormat_D24_UNorm_S8_UInt: return Format{ GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 };
  case FG_PixelFormat_X24_Typeless_G8_UInt: return Format{ 0, 0, 0 };
  case FG_PixelFormat_R8G8_Typeless: return Format{ GL_RG, GL_RG, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8_UNorm: return Format{ GL_RG8, GL_RG, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8_UInt: return Format{ GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8G8_SNorm: return Format{ GL_RG8_SNORM, GL_RG, GL_BYTE };
  case FG_PixelFormat_R8G8_Int: return Format{ GL_RG8I, GL_RG_INTEGER, GL_BYTE };
  case FG_PixelFormat_R16_Typeless: return Format{ GL_R16, GL_RED, GL_HALF_FLOAT };
  case FG_PixelFormat_R16_Float: return Format{ GL_R16F, GL_RED, GL_HALF_FLOAT };
  case FG_PixelFormat_R16_UNorm: return Format{ GL_R16, GL_RED, GL_SHORT };
  case FG_PixelFormat_R16_UInt: return Format{ GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R16_SNorm: return Format{ GL_R16_SNORM, GL_RED, GL_SHORT };
  case FG_PixelFormat_R16_Int: return Format{ GL_R16I, GL_RED_INTEGER, GL_SHORT };
  case FG_PixelFormat_R8_Typeless: return Format{ GL_RED, GL_RED, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8_UNorm: return Format{ GL_RED, GL_RED, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R8_UInt: return Format{ GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT };
  case FG_PixelFormat_R8_SNorm: return Format{ GL_R8_SNORM, GL_RED, GL_BYTE };
  case FG_PixelFormat_R8_Int: return Format{ GL_R16I, GL_RED_INTEGER, GL_SHORT };
  case FG_PixelFormat_A8_UNorm: return Format{ GL_ALPHA8, GL_ALPHA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R1_UNorm: return Format{ 0, 0, 0 };
  case FG_PixelFormat_R9G9B9E5_SHAREDEXP: return Format{ 0, 0, 0 };
  case FG_PixelFormat_R8G8_B8G8_UNorm: return Format{ 0, 0, 0 };
  case FG_PixelFormat_G8R8_G8B8_UNorm: return Format{ 0, 0, 0 };
  case FG_PixelFormat_R5G6B5_UNorm: return Format{ GL_RGB, GL_RGB, GL_UNSIGNED_SHORT_5_6_5 };
  case FG_PixelFormat_R5G5B5A1_UNorm: return Format{ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1 };
  case FG_PixelFormat_B5G6R5_UNorm: return Format{ GL_BGR, GL_BGR, GL_UNSIGNED_SHORT_5_6_5_REV };
  case FG_PixelFormat_B5G5R5A1_UNorm: return Format{ GL_BGRA, GL_BGRA, GL_UNSIGNED_SHORT_1_5_5_5_REV };
  case FG_PixelFormat_B8G8R8A8_UNorm: return Format{ GL_BGRA, GL_BGRA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_B8G8R8X8_UNorm: return Format{ GL_BGR, GL_BGR, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_R10G10B10_XR_BIAS_A2_UNorm: return Format{ 0, 0, 0 };
  case FG_PixelFormat_B8G8R8A8_Typeless: return Format{ GL_BGRA, GL_BGRA, GL_UNSIGNED_BYTE };
  case FG_PixelFormat_B8G8R8X8_Typeless: return Format{ GL_BGR, GL_BGR, GL_UNSIGNED_BYTE };
  }
  return Format{ 0, 0, 0 };
}
