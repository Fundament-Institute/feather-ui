local C = terralib.includecstring [[
  #include <string.h>
]]

local B = {}
local F = require 'feather.shared'
local Flags = require 'feather.flags'
local Enum = require 'feather.enum'
local OS = require 'feather.os'
local Alloc = require 'std.alloc'
local Msg = require 'feather.message'
local Log = require 'feather.log'
local C = require 'feather.libc'
local tunpack = table.unpack or unpack

B.Feature = Flags{
  "TEXT_ANTIALIAS", 
  "TEXT_SUBPIXEL",
  "TEXT_BLUR",
  "TEXT_ALPHA",
  "SHAPE_BLUR",
  "SHAPE_ALPHA",
  "RECT_CORNERS",
  "RECT_BORDER",
  "CIRCLE_INNER",
  "CIRCLE_BORDER",
  "ARC_INNER",
  "ARC_BORDER",
  "TRIANGLE_CORNERS",
  "TRIANGLE_BORDER",
  "LINES_ALPHA",
  "CURVE_STROKE", -- If both stroke and fill are false, it doesn't support curves at all
  "CURVE_FILL",
  "LAYER_TRANSFORM",
  "LAYER_OPACITY",
  "GAMMA",
  "BATCHING", -- If true, the backend is capable of batching certain command lists together. 
  "SHADER_GLSL_ES2", -- Shader type determines what CreateShader will accept
  "SHADER_GLSL4",
  "SHADER_HLSL2",
  "BACKGROUND_OPACITY",
  "IMMEDIATE_MODE"}

B.Format = Enum{
  "GRADIENT",
  "VERTEX_BUFFER",
  "ELEMENT_BUFFER",
  "UNIFORM_BUFFER",
  "STORAGE_BUFFER",
  "UNKNOWN_BUFFER",
  "LAYER",
  "WEAK_LAYER",
  "TEXTURE",
  "ATLAS",
  "BMP",
  "JPG",
  "PNG",
  "ICO",
  "GIF",
  "TIFF",
  "TGA",
  "WEBP",
  "DDS",
  "WIC",
  "SVG",
  "AVI",
  "MP4",
  "MKV",
  "WEBM"}

B.Format.methods["UNKNOWN"] = constant(`0xff)
B.Format.enum_values["UNKNOWN"] = 0xff

B.AntiAliasing = Flags{
  "AA",
  "LCD",
  "LCD_V",
  "SDF",
}

B.BreakStyle = Enum{
  "NONE",
  "WORD",
  "CHARACTER",
}

B.Clipboard = Enum{
  "NONE",
  "TEXT",
  "WAVE",
  "BITMAP",
  "FILE",
  "ELEMENT",
  "CUSTOM",
  "ALL",
}

B.Cursor = Enum{
  "NONE",
  "ARROW",
  "IBEAM",
  "CROSS",
  "WAIT",
  "HAND",
  "RESIZENS",
  "RESIZEWE",
  "RESIZENWSE",
  "RESIZENESW",
  "RESIZEALL",
  "NO",
  "HELP",
  "DRAG",
  "CUSTOM",
}


B.DrawFlags = Flags({
  "CCW_FRONT_FACE",
  "CULL_FACE",
  "WIREFRAME",
  "POINTMODE",
  "INSTANCED",
  "LINEAR",
}, uint8)

struct B.Data {
  union {
    data : &opaque
    index : uint
  }
}

struct B.Font {
  data : B.Data
  dpi : F.Vec
  baseline : float
  lineheight : float
  pt : uint
  aa : B.AntiAliasing
}

B.PixelFormat = Enum({
  "UNKNOWN",
  "R32G32B32A32_TYPELESS",
  "R32G32B32A32_FLOAT",
  "R32G32B32A32_UINT",
  "R32G32B32A32_INT",
  "R32G32B32_TYPELESS",
  "R32G32B32_FLOAT",
  "R32G32B32_UINT",
  "R32G32B32_INT",
  "R16G16B16A16_TYPELESS",
  "R16G16B16A16_FLOAT",
  "R16G16B16A16_UNORM",
  "R16G16B16A16_UINT",
  "R16G16B16A16_SNORM",
  "R16G16B16A16_INT",
  "R16G16B16_TYPELESS",
  "R16G16B16_FLOAT",
  "R16G16B16_UNORM",
  "R16G16B16_UINT",
  "R16G16B16_SNORM",
  "R16G16B16_INT",
  "R32G32_TYPELESS",
  "R32G32_FLOAT",
  "R32G32_UINT",
  "R32G32_INT",
  "R32G8X24_TYPELESS",
  "D32_FLOAT_S8X24_UINT",
  "R32_FLOAT_X8X24_TYPELESS",
  "X32_TYPELESS_G8X24_UINT",
  "R10G10B10A2_TYPELESS",
  "R10G10B10A2_UNORM",
  "R10G10B10A2_UINT",
  "R11G11B10_FLOAT",
  "R8G8B8A8_TYPELESS",
  "R8G8B8A8_UNORM",
  "R8G8B8A8_UNORM_SRGB",
  "R8G8B8A8_UINT",
  "R8G8B8A8_SNORM",
  "R8G8B8A8_INT",
  "R8G8B8X8_TYPELESS",
  "R8G8B8X8_UNORM",
  "R8G8B8X8_SNORM",
  "R8G8B8X8_UINT",
  "R8G8B8X8_INT",
  "R16G16_TYPELESS",
  "R16G16_FLOAT",
  "R16G16_UNORM",
  "R16G16_UINT",
  "R16G16_SNORM",
  "R16G16_INT",
  "R32_TYPELESS",
  "D32_FLOAT",
  "R32_FLOAT",
  "R32_UINT",
  "R32_INT",
  "R24G8_TYPELESS",
  "D24_UNORM_S8_UINT",
  "D24_UNORM_X8_TYPELESS",
  "R24_UNORM_X8_TYPELESS",
  "X24_TYPELESS_G8_UINT",
  "R8G8_TYPELESS",
  "R8G8_UNORM",
  "R8G8_UINT",
  "R8G8_SNORM",
  "R8G8_INT",
  "R16_TYPELESS",
  "R16_FLOAT",
  "D16_UNORM",
  "R16_UNORM",
  "R16_UINT",
  "R16_SNORM",
  "R16_INT",
  "R8_TYPELESS",
  "R8_UNORM",
  "R8_UINT",
  "R8_SNORM",
  "R8_INT",
  "A8_UNORM",
  "R1_UNORM",
  "R9G9B9E5_SHAREDEXP",
  "R8G8_B8G8_UNORM",
  "G8R8_G8B8_UNORM",
  "BC1_TYPELESS",
  "BC1_UNORM",
  "BC1_UNORM_SRGB",
  "BC2_TYPELESS",
  "BC2_UNORM",
  "BC2_UNORM_SRGB",
  "BC3_TYPELESS",
  "BC3_UNORM",
  "BC3_UNORM_SRGB",
  "BC4_TYPELESS",
  "BC4_UNORM",
  "BC4_SNORM",
  "BC5_TYPELESS",
  "BC5_UNORM",
  "BC5_SNORM",
  "R5G6B5_UNORM",
  "R5G5B5A1_UNORM",
  "B5G6R5_UNORM",
  "B5G5R5A1_UNORM",
  "B8G8R8A8_UNORM",
  "B8G8R8X8_UNORM",
  "R10G10B10_XR_BIAS_A2_UNORM",
  "B8G8R8A8_TYPELESS",
  "B8G8R8A8_UNORM_SRGB",
  "B8G8R8X8_TYPELESS",
  "B8G8R8X8_UNORM_SRGB",
  "BC6H_TYPELESS",
  "BC6H_UF16",
  "BC6H_SF16",
  "BC7_TYPELESS",
  "BC7_UNORM",
  "BC7_UNORM_SRGB",
  "AYUV",
  "Y410",
  "Y416",
  "NV12",
  "P010",
  "P016",
  "YUY2",
  "Y210",
  "Y216",
  "NV11",
  "AI44",
  "IA44",
  "P8",
  "A8P8",
  "B4G4R4A4_UNORM",
  "P208",
  "V208",
  "V408"
}, uint8)

B.Primitive = Enum({
  "POINT",
  "LINE",
  "TRIANGLE",
  "LINE_STRIP",
  "TRIANGLE_STRIP",
  "LINE_ADJACENCY",
  "TRIANGLE_ADJACENCY",
  "LINE_STRIP_ADJACENCY",
  "TRIANGLE_STRIP_ADJACENCY",
}, uint8)

B.ShaderType = Enum{
  "HALF",
  "FLOAT",
  "DOUBLE",
  "INT",
  "UINT",
  "COLOR32",
  "TEXTURE",
  "TEXCUBE",
}

struct B.ShaderParameter {
  type : B.ShaderType
  length : uint -- for arrays
  multi : uint -- for matrices, or to indicate TEX1D, TEX2D, TEX3D
  name : F.conststring
  index : uint16 -- says what buffer index this is for
  step : uint16 -- instanced data step
}

B.AssetFlags = Flags({
  "LINEAR",
  "NO_MIPMAP",
  "CACHE_LAYER",
  "CUBE_MAP",
})

struct B.Asset {
  data : B.Data
  format : B.Format
  flags : B.AssetFlags
  union {
    texture : struct {
      size : F.Veci
      dpi : F.Vec
    }
    resource : struct {
      bytes : uint
      subformat : B.PixelFormat
    }
  }
}
B.Asset.c_export = [[FG_Data data;
  enum FG_Format format;
  int flags;
  union {
    struct {
      FG_Veci size;
      FG_Vec dpi;
    };
    struct {
      uint32_t bytes;
      uint8_t subformat;
    };
  };]]

struct B.ShaderValue {
  union {
    f32 : float
    f64 : double
    i32 : int
    u32 : uint
    pf32 : &float
    pf64 : &double
    pi32 : &int
    pu32 : &uint
    asset : &B.Asset
  }
}

struct B.Shader {
  data : &opaque
  parameters : &B.ShaderParameter
  n_parameters : uint
}

struct B.Display {
  size : F.Veci
  offset : F.Veci
  dpi : F.Vec
  scale : float
  handle : &opaque
  primary : bool
}

B.BlendValue = Enum({
  "ZERO", 
  "ONE", 
  "SRC_COLOR", 
  "INV_SRC_COLOR",
  "DST_COLOR", 
  "INV_DST_COLOR", 
  "SRC_ALPHA",
  "INV_SRC_ALPHA",
  "DST_ALPHA", 
  "INV_DST_ALPHA",
  "CONSTANT_COLOR", 
  "INV_CONSTANT_COLOR",
  "CONSTANT_ALPHA", 
  "INV_CONSTANT_ALPHA",
  "SRC_ALPHA_SATURATE",
}, uint8)

B.BlendOp = Enum({
  "ADD",
  "SUBTRACT",
  "REV_SUBTRACT",
}, uint8)

struct B.BlendState {
  srcBlend : B.BlendValue
  destBlend : B.BlendValue
  colorBlend : B.BlendOp
  srcBlendAlpha : B.BlendValue
  destBlendAlpha : B.BlendValue
  alphaBlend : B.BlendOp
  mask : uint8 -- RGBA mask, R is bit 1, A is bit 4
  flags : B.DrawFlags
  constant : F.Color
}

B.Category = Enum({
  "TEXT",
  "ASSET",
  "RECT",
  "CIRCLE",
  "ARC",
  "TRIANGLE",
  "LINES",
  "CURVE",
  "LINES3D",
  "CURVE3D",
  "CUBE",
  "ICOSPHERE",
  "CYLINDER",
  "SHADER"
}, uint8)

struct B.Command {
  category : B.Category
  union {
    text : struct { 
      font : &B.Font, 
      layout : &opaque, 
      area : &F.Rect, 
      color : F.Color,
      blur : float, 
      rotate : float, 
      z : float 
    }
    asset : struct { 
      asset : &B.Asset, 
      area : &F.Rect, 
      source : &F.Rect, 
      color : F.Color, 
      time : float, 
      rotate : float, 
      z : float 
    }
    shape : struct {
      area : &F.Rect, 
      fillColor : F.Color
      border : float
      borderColor : F.Color
      blur : float, 
      asset : &B.Asset
      z : float
      union {
        rect : struct {
          corners : &F.Rect, 
          rotate : float, 
        }
        circle : struct {
          innerRadius : float
          innerBorder : float
        }
        arc : struct {
          angles : F.Vec
          innerRadius : float
        }
        triangle : struct {
          corners : &F.Rect
          rotate : float
        }
      }
    }
    lines : struct {
      union {
        points : &F.Vec
        points3D : &F.Vec3
      }
      count : uint
      color : F.Color
    }
    curve : struct {
      union {
        points : &F.Vec
        points3D : &F.Vec3
      }
      count : uint
      fillColor : F.Color
      stroke : float
      strokeColor : F.Color
    }
    shape3D : struct {
      shader : &B.Shader
      subdivision : int
      values : &B.ShaderValue
    }
    shader : struct {
      shader : &B.Shader
      primitive : B.Primitive
      input : &opaque
      count : uint
      values : &B.ShaderValue
    }
  }
}

struct B.Backend {
  destroy : {&B.Backend} -> {}

  features : B.Feature
  formats : uint
  dpi : F.Vec
  scale : float
  cursorblink : uint64
  tooltipdelay : uint64
}

-- Define a dynamic backend object (a static backend would be a seperate type that also provides these functions).
terra B.Backend:Draw(window : &Msg.Window, commands : &B.Command, n_commands : uint, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:Clear(window : &Msg.Window, color : F.Color) : bool return false end -- Clears whatever is inside the current clipping rect
terra B.Backend:PushLayer(window : &Msg.Window, layer : &B.Asset, transform : &float, opacity : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:PopLayer(window : &Msg.Window) : F.Err return 0 end
terra B.Backend:PushRenderTarget(window : &Msg.Window, target : &B.Asset) : F.Err return 0 end
terra B.Backend:PopRenderTarget(window : &Msg.Window) : F.Err return 0 end
terra B.Backend:PushClip(window : &Msg.Window, area : &F.Rect) : F.Err return 0 end
terra B.Backend:PopClip(window : &Msg.Window) : F.Err return 0 end
terra B.Backend:DirtyRect(window : &Msg.Window, area : &F.Rect) : F.Err return 0 end
terra B.Backend:BeginDraw(window : &Msg.Window, area : &F.Rect) : F.Err return 0 end
terra B.Backend:EndDraw(window : &Msg.Window) : F.Err return 0 end

terra B.Backend:CreateShader(ps : F.conststring, vs : F.conststring, gs : F.conststring, cs : F.conststring, ds : F.conststring, hs : F.conststring, parameters : &B.ShaderParameter, n_parameters : uint) : &B.Shader return nil end
terra B.Backend:DestroyShader(shader : &B.Shader) : F.Err return 0 end
terra B.Backend:CreateShaderInput(buffers : &&B.Asset, n_buffers : uint, parameters : &B.ShaderParameter, n_parameters : uint) : &opaque return nil end
terra B.Backend:DestroyShaderInput(input : &opaque) : F.Err return 0 end

terra B.Backend:CreateFont(family : F.conststring, weight : uint16, italic : bool, pt : uint32, dpi : F.Vec, aa : B.AntiAliasing) : &B.Font return nil end
terra B.Backend:DestroyFont(font : &B.Font) : F.Err return 0 end
terra B.Backend:FontLayout(font : &B.Font, text : F.conststring, area : &F.Rect, lineHeight : float, letterSpacing : float, breakStyle : B.BreakStyle, previous : &opaque) : &opaque return nil end
terra B.Backend:DestroyLayout(layout : &opaque) : F.Err return 0 end
terra B.Backend:FontIndex(font : &B.Font, layout : &opaque, area : &F.Rect, pos : F.Vec, cursor : &F.Vec) : uint return 0 end
terra B.Backend:FontPos(font : &B.Font, layout : &opaque, area :&F.Rect, index : uint) : F.Vec return F.Vec{} end

terra B.Backend:CreateAsset(window : &Msg.Window, data : F.conststring, count : uint, format : B.Format, flags : int) : &B.Asset return nil end
terra B.Backend:CreateBuffer(window : &Msg.Window, data : &opaque, bytes : uint, format : B.PixelFormat, type : B.Format) : &B.Asset return nil end
terra B.Backend:CreateLayer(window : &Msg.Window, size : &F.Vec, flags : int) : &B.Asset return nil end
terra B.Backend:CreateRenderTarget(window : &Msg.Window, size : &F.Vec, format : B.PixelFormat, flags : int) : &B.Asset return nil end
--terra B.Backend:CreateAtlas(assets : &B.Asset, count : uint, flags : int) : &B.Asset return nil end
--terra B.Backend:AddAtlas(atlas : &B.Asset, assets : &B.Asset, count : uint, flags : int) : F.Err return 0 end
--terra B.Backend:RemoveAtlas(atlas : &B.Asset, assets : &B.Asset, count : uint, flags : int) : F.Err return 0 end
--terra B.Backend:AsyncLoadAsset(window : &Msg.Window, data : F.conststring, count : uint, format : B.Format, flags : int) : &B.Asset return nil end
terra B.Backend:PrefetchAsset(asset: &B.Asset, time : float) : F.Err return 0 end
--terra B.Backend:QueryAssetStatus(asset: &B.Asset, time : float) : F.Err return 0 end
terra B.Backend:DestroyAsset(asset : &B.Asset) : F.Err return 0 end
terra B.Backend:GetProjection(window : &Msg.Window, layer : &B.Asset, proj4x4 : &float) : F.Err return 0 end

terra B.Backend:PutClipboard(window : &Msg.Window, kind : B.Clipboard, data : F.conststring, count : uint) : F.Err return 0 end
terra B.Backend:GetClipboard(window : &Msg.Window, kind : B.Clipboard, target : &opaque, count : uint) : uint return 0 end
terra B.Backend:CheckClipboard(window : &Msg.Window, kind : B.Clipboard) : bool return false end
terra B.Backend:ClearClipboard(window : &Msg.Window, kind : B.Clipboard) : F.Err return 0 end

terra B.Backend:CreateSystemControl(window : &Msg.Window, id : F.conststring, area : &F.Rect, ...) : &opaque return nil end
terra B.Backend:SetSystemControl(window : &Msg.Window, control : &opaque, area : &F.Rect, ...) : F.Err return 0 end
terra B.Backend:DestroySystemControl(window : &Msg.Window, control : &opaque) : F.Err return 0 end

terra B.Backend:GetSyncObject() : &opaque return nil end
terra B.Backend:ProcessMessages() : F.Err return 0 end
terra B.Backend:SetCursor(window : &Msg.Window, cursor : B.Cursor) : F.Err return 0 end
terra B.Backend:GetDisplayIndex(index : uint, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplay(handle : &opaque, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplayWindow(window : &Msg.Window, out : &B.Display) : F.Err return 0 end
terra B.Backend:CreateRegion(element : &Msg.Receiver, window : Msg.Window, pos : F.Vec3, dim : F.Vec3) : &Msg.Window return nil end
terra B.Backend:CreateWindow(element : &Msg.Receiver, display : &opaque, pos : &F.Vec, dim : &F.Vec, caption : F.conststring, flags : uint64) : &Msg.Window return nil end
terra B.Backend:SetWindow(window : &Msg.Window, element : &Msg.Receiver, display : &opaque, pos : &F.Vec, dim : &F.Vec, caption : F.conststring, flags : uint64) : F.Err return 0 end
terra B.Backend:DestroyWindow(window : &Msg.Window) : F.Err return 0 end

-- Generate both the function pointer field and redefine the function to call the generated function pointer
do
  local map = {}
  for k, v in pairs(B.Backend.methods) do
    klow = string.lower(k:sub(1,1)) .. k:sub(2, -1)
    v.c_body = "return (*self->"..klow..")("

    for i, p in ipairs(v.definition.parameters) do
      if i > 1 then 
        v.c_body = v.c_body .. ", "
      end
      v.c_body = v.c_body .. p.name
    end
    
    v.c_body = v.c_body .. ");"

    v.type.parameters:map(symbol)
    B[k] = v
    B.Backend.entries:insert({field = klow, type = terralib.types.pointer(v:gettype())})
    
    map[klow] = v -- TODO: alphabetically sort map before putting in entries, or use a C struct sorting method on the resulting struct
  end

  for k, v in pairs(map) do
    local args = v.definition.parameters:map(function(x) return x.symbol end)
    if v.type.isvararg then
      v:resetdefinition(terra([args], ...): v.type.returntype return [args[1]].[k]([args]) end)
    else
      v:resetdefinition(terra([args]): v.type.returntype return [args[1]].[k]([args]) end)
    end
  end
end

terra B.Backend:free(library : &opaque) : {}
  self.destroy(self)
  if library ~= nil then 
    OS.FreeLibrary(library)
  end
end

B.Log = terralib.types.funcpointer({&opaque, Log.Level, F.conststring}, {}, true)
B.InitBackend = {&opaque, B.Log, Msg.Behavior} -> &B.Backend

terra LoadDynamicBackend(ui : &opaque, behavior : Msg.Behavior, log : B.Log, path : rawstring, name : rawstring) : {&B.Backend, &opaque}
  var l : &opaque = OS.LoadLibrary(path)

  if l == nil then
    C.printf("Error loading %s: %s\n", path, OS.LoadDLLError())
    return nil, nil
  end

  var str : rawstring = nil
  if name == nil then
    name = C.strrchr(path, ("\\")[0])
    if name == nil then
      name = C.strrchr(path, ("/")[0])
    end
    name = terralib.select(name == nil, path, name + 1)

    if C.strncmp(name, "lib", 3) == 0 then
      name = name + 3
    end

    str = Alloc.default_allocator:alloc(int8, C.strlen(name) + 1)
    C.strcpy(str, name)
    name = str

    var ext : rawstring = C.strrchr(str, (".")[0])
    if ext ~= nil then
      @ext = 0
    end

    -- strip _d from the name
    var sub : rawstring = C.strrchr(str, ("_")[0])
    if sub ~= nil and sub[1] ~= 0 and sub[1] == ("d")[0] then
      @sub = 0
    end
  end

  var f : B.InitBackend = [B.InitBackend](OS.LoadFunction(l, name))

  defer [terra (s : rawstring) : {} if s ~= nil then Alloc.free(s) end end](str)

  if f ~= nil then
    var b : &B.Backend = f(ui, log, behavior)
    if b ~= nil then
      return b, l
    end
    C.printf("Init function failed in %s", path)
  else
    C.printf("Can't find init function %s() in %s", name, path)
  end

  OS.FreeLibrary(l)
  return nil, nil
end

-- In cases where the backend is not known at compile time, we must load it via a shared library at runtime 
terra B.Backend.methods.new(ui : &opaque, behavior : Msg.Behavior, log : B.Log, path : rawstring, name : rawstring) : {&B.Backend, &opaque}
  return LoadDynamicBackend(ui, behavior, log, path, name)
end

return B