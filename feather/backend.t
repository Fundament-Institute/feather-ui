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

B.Feature = Flags{
  "TEXT_ANTIALIAS", 
  "TEXT_SUBPIXEL",
  "TEXT_BLUR",
  "TEXT_ALPHA",
  "RECT_CORNERS",
  "RECT_BORDER",
  "RECT_BLUR",
  "RECT_ALPHA",
  "CIRCLE_ARCS",
  "CIRCLE_BORDER",
  "CIRCLE_BLUR",
  "CIRCLE_ALPHA",
  "TRIANGLE_CORNERS",
  "TRIANGLE_BORDER",
  "TRIANGLE_BLUR",
  "TRIANGLE_ALPHA",
  "LINES_ALPHA",
  "CURVE_STROKE", -- If both stroke and fill are false, it doesn't support curves at all
  "CURVE_FILL",
  "LAYER_TRANSFORM",
  "LAYER_OPACITY",
  "SHADER_GLSL_ES2", -- Shader type determines what CreateShader will accept
  "SHADER_GLSL4",
  "SHADER_HLSL2",
  "BACKGROUND_OPACITY",
  "IMMEDIATE_MODE"}

B.Format = Enum{
  "GRADIENT",
  "BUFFER",
  "LAYER",
  "WEAK_LAYER",
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

B.AntiAliasing = Enum{
  "NO_AA",
  "AA",
  "LCD",
  "LCD_V"}

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
  "ALL"}

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
  "CUSTOM"}

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

B.Primitive = Enum({
  "POINT",
  "LINE",
  "TRIANGLE",
  "LINE_STRIP",
  "TRIANGLE_STRIP",
  "INDEX_BYTE",
  "INDEX_SHORT",
  "INDEX_INT",
}, uint8)

struct B.Asset {
  data : B.Data
  format : B.Format
  size : F.Veci
  dpi : F.Vec
}
B.Asset.c_export = [[FG_Data data;
  FG_Format format;
  union {
    struct {
      FG_Veci size;
      FG_Vec dpi;
    };
    struct {
      unsigned int count;
      unsigned short stride;
      unsigned char primitive;
      FG_ShaderParameter * parameters;
      unsigned int n_parameters;
    };
  };]]

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
  constant : F.Color
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
terra B.Backend:DrawText(window : &opaque, font : &B.Font, layout : &opaque, area : &F.Rect, color : F.Color, blur : float, rotate : float, z : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawAsset(window : &opaque, asset : &B.Asset, area : &F.Rect, source : &F.Rect, color : F.Color, time : float, rotate : float, z : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawRect(window : &opaque, area : &F.Rect, corners : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset, rotate : float, z : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawCircle(window : &opaque, area : &F.Rect, angles : F.Vec, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, innerRadius : float, innerBorder : float, asset : &B.Asset, z : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawTriangle(window : &opaque, area : &F.Rect, corners : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset, rotate : float, z : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawLines(window : &opaque, points : &F.Vec, count : uint, color : F.Color, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawCurve(window : &opaque, anchors : &F.Vec, count : uint, fillColor : F.Color, stroke : float, strokeColor : F.Color, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:DrawShader(window : &opaque, shader : &B.Shader, vertices : &B.Asset, indices : &B.Asset, blendstate : &B.BlendState, ...) : F.Err return 0 end

terra B.Backend:Clear(window : &opaque, color : F.Color) : bool return false end -- Clears whatever is inside the current clipping rect
terra B.Backend:PushLayer(window : &opaque, layer : &B.Asset, transform : &float, opacity : float, blendstate : &B.BlendState) : F.Err return 0 end
terra B.Backend:PopLayer(window : &opaque) : F.Err return 0 end
terra B.Backend:SetRenderTarget(window : &opaque, target : &B.Asset) : F.Err return 0 end
terra B.Backend:PushClip(window : &opaque, area : &F.Rect) : F.Err return 0 end
terra B.Backend:PopClip(window : &opaque) : F.Err return 0 end
terra B.Backend:DirtyRect(window : &opaque, area : &F.Rect) : F.Err return 0 end
terra B.Backend:BeginDraw(window : &opaque, area : &F.Rect) : F.Err return 0 end
terra B.Backend:EndDraw(window : &opaque) : F.Err return 0 end

terra B.Backend:CreateShader(ps : F.conststring, vs : F.conststring, gs : F.conststring, cs : F.conststring, ds : F.conststring, hs : F.conststring, parameters : &B.ShaderParameter, n_parameters : uint) : &B.Shader return nil end
terra B.Backend:DestroyShader(shader : &B.Shader) : F.Err return 0 end

terra B.Backend:CreateFont(family : F.conststring, weight : uint16, italic : bool, pt : uint32, dpi : F.Vec, aa : B.AntiAliasing) : &B.Font return nil end
terra B.Backend:DestroyFont(font : &B.Font) : F.Err return 0 end
terra B.Backend:FontLayout(font : &B.Font, text : F.conststring, area : &F.Rect, lineHeight : float, letterSpacing : float, breakStyle : B.BreakStyle, previous : &opaque) : &opaque return nil end
terra B.Backend:DestroyLayout(layout : &opaque) : F.Err return 0 end
terra B.Backend:FontIndex(font : &B.Font, layout : &opaque, area : &F.Rect, pos : F.Vec, cursor : &F.Vec) : uint return 0 end
terra B.Backend:FontPos(font : &B.Font, layout : &opaque, area :&F.Rect, index : uint) : F.Vec return F.Vec{} end

terra B.Backend:CreateAsset(data : F.conststring, count : uint, format : B.Format) : &B.Asset return nil end
terra B.Backend:CreateBuffer(data : &opaque, bytes : uint, primitive : B.Primitive, parameters : &B.ShaderParameter, n_parameters : uint) : &B.Asset return nil end
terra B.Backend:CreateLayer(window : &opaque, size : &F.Vec, cache : bool) : &B.Asset return nil end
terra B.Backend:DestroyAsset(asset : &B.Asset) : F.Err return 0 end
terra B.Backend:GetProjection(window : &opaque, layer : &opaque, proj4x4 : &float) : F.Err return 0 end

terra B.Backend:PutClipboard(window : &opaque, kind : B.Clipboard, data : F.conststring, count : uint) : F.Err return 0 end
terra B.Backend:GetClipboard(window : &opaque, kind : B.Clipboard, target : &opaque, count : uint) : uint return 0 end
terra B.Backend:CheckClipboard(window : &opaque, kind : B.Clipboard) : bool return false end
terra B.Backend:ClearClipboard(window : &opaque, kind : B.Clipboard) : F.Err return 0 end

terra B.Backend:CreateSystemControl(window : &opaque, id : F.conststring, area : &F.Rect, ...) : &opaque return nil end
terra B.Backend:SetSystemControl(window : &opaque, control : &opaque, area : &F.Rect, ...) : F.Err return 0 end
terra B.Backend:DestroySystemControl(window : &opaque, control : &opaque) : F.Err return 0 end

terra B.Backend:GetSyncObject() : &opaque return nil end
terra B.Backend:ProcessMessages() : F.Err return 0 end
terra B.Backend:SetCursor(window : &opaque, cursor : B.Cursor) : F.Err return 0 end
terra B.Backend:GetDisplayIndex(index : uint, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplay(handle : &opaque, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplayWindow(window : &opaque, out : &B.Display) : F.Err return 0 end
terra B.Backend:CreateWindow(element : &Msg.Receiver, display : &opaque, pos : &F.Vec, dim : &F.Vec, caption : F.conststring, flags : uint64, context : &opaque) : &opaque return nil end
terra B.Backend:SetWindow(window : &opaque, element : &Msg.Receiver, display : &opaque, pos : &F.Vec, dim : &F.Vec, caption : F.conststring, flags : uint64) : F.Err return 0 end
terra B.Backend:DestroyWindow(window : &opaque) : F.Err return 0 end

-- Generate both the function pointer field and redefine the function to call the generated function pointer
do
  local map = {}
  for k, v in pairs(B.Backend.methods) do
    k = string.lower(k:sub(1,1)) .. k:sub(2, -1)
    B.Backend.entries:insert({field = k, type = terralib.types.pointer(v:gettype())})
    map[k] = v -- TODO: alphabetically sort map before putting in entries, or use a C struct sorting method on the resulting struct
  end

  for k, v in pairs(map) do
    local args = v.type.parameters:map(symbol)
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