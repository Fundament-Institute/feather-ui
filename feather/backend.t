local C = terralib.includecstring [[
  #include <string.h>
]]

local B = {}
local F = require 'feather.shared'
local Flags = require 'feather.flags'
local Enum = require 'feather.enum'
local Element = require 'feather.element'
local OS = require 'feather.os'
local Alloc = require 'std.alloc'
local M = require 'feather.message'
local L = require 'feather.log'
local U = require 'feather.util'
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
  "SHADER_GLSL2",
  "SHADER_GLSL4",
  "SHADER_HLSL2",
  "BACKGROUND_OPACITY",
  "IMMEDIATE_MODE"}

B.Format = Enum{
  "GRADIENT",
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
  data : &opaque
}

struct B.Font {
  data : B.Data
  dpi : F.Vec
  baseline : float
  lineheight : float
  pt : uint
  aa : B.AntiAliasing
}

struct B.Asset {
  data : B.Data
  size : F.Veci
  dpi : F.Vec
  format : B.Format
}

struct B.Display {
  size : F.Veci
  offset : F.Veci
  dpi : F.Vec
  scale : float
  handle : &opaque
  primary : bool
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
terra B.Backend:DrawText(window : &opaque, font : &B.Font, layout : &opaque, area : &F.Rect, color : F.Color, blur : float) : F.Err return 0 end
terra B.Backend:DrawAsset(window : &opaque, asset : &B.Asset, area : &F.Rect, source : &F.Rect, color : F.Color, time : float) : F.Err return 0 end
terra B.Backend:DrawRect(window : &opaque, area : &F.Rect, corners : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset) : F.Err return 0 end
terra B.Backend:DrawCircle(window : &opaque, area : &F.Rect, arcs : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset) : F.Err return 0 end
terra B.Backend:DrawTriangle(window : &opaque, area : &F.Rect, corners : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset) : F.Err return 0 end
terra B.Backend:DrawLines(window : &opaque, points : &F.Vec, count : uint, color : F.Color) : F.Err return 0 end
terra B.Backend:DrawCurve(window : &opaque, anchors : &F.Vec, count : uint, fillColor : F.Color, stroke : float, strokeColor : F.Color) : F.Err return 0 end
terra B.Backend:PushLayer(window : &opaque, area : &F.Rect, transform : &float, opacity : float, cache : &opaque) : F.Err return 0 end
terra B.Backend:PopLayer(window : &opaque) : &opaque return nil end
terra B.Backend:DestroyLayer(window : &opaque, layer : &opaque) : F.Err return 0 end
terra B.Backend:PushClip(window : &opaque, area : &F.Rect) : F.Err return 0 end
terra B.Backend:PopClip(window : &opaque) : F.Err return 0 end
terra B.Backend:DirtyRect(window : &opaque, layer : &opaque, area : &F.Rect) : F.Err return 0 end
terra B.Backend:BeginDraw(window : &opaque, area : &F.Rect, clear : bool) : F.Err return 0 end
terra B.Backend:EndDraw(window : &opaque) : F.Err return 0 end

terra B.Backend:CreateFont(family : F.conststring, weight : uint16, italic : bool, pt : uint32, dpi : F.Vec, aa : B.AntiAliasing) : &B.Font return nil end
terra B.Backend:DestroyFont(font : &B.Font) : F.Err return 0 end
terra B.Backend:FontLayout(font : &B.Font, text : F.conststring, area : &F.Rect, lineHeight : float, letterSpacing : float, breakStyle : B.BreakStyle, previous : &opaque) : &opaque return nil end
terra B.Backend:DestroyLayout(layout : &opaque) : F.Err return 0 end
terra B.Backend:FontIndex(font : &B.Font, layout : &opaque, area : &F.Rect, pos : F.Vec, cursor : &F.Vec) : uint return 0 end
terra B.Backend:FontPos(font : &B.Font, layout : &opaque, area :&F.Rect, index : uint) : F.Vec return F.Vec{} end

terra B.Backend:CreateAsset(data : F.conststring, count : uint, format : B.Format) : &B.Asset return nil end
terra B.Backend:DestroyAsset(asset : &B.Asset) : F.Err return 0 end

terra B.Backend:PutClipboard(window : &opaque, kind : B.Clipboard, data : F.conststring, count : uint) : F.Err return 0 end
terra B.Backend:GetClipboard(window : &opaque, kind : B.Clipboard, target : &opaque, count : uint) : uint return 0 end
terra B.Backend:CheckClipboard(window : &opaque, kind : B.Clipboard) : bool return false end
terra B.Backend:ClearClipboard(window : &opaque, kind : B.Clipboard) : F.Err return 0 end

terra B.Backend:GetSyncObject() : &opaque return nil end
terra B.Backend:ProcessMessages() : F.Err return 0 end
terra B.Backend:SetCursor(window : &opaque, cursor : B.Cursor) : F.Err return 0 end
terra B.Backend:RequestAnimationFrame(window : &opaque, delay : uint64) : F.Err return 0 end
terra B.Backend:GetDisplayIndex(index : uint, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplay(handle : &opaque, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplayWindow(window : &opaque, out : &B.Display) : F.Err return 0 end
terra B.Backend:CreateWindow(element : &Element, display : &opaque, pos : &F.Vec, dim : &F.Vec, caption : F.conststring, flags : uint64, context : &opaque) : &opaque return nil end
terra B.Backend:SetWindow(window : &opaque, element : &Element, display : &opaque, pos : &F.Vec, dim : &F.Vec, caption : F.conststring, flags : uint64) : F.Err return 0 end
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
    v:resetdefinition(terra([args]): v.type.returntype return [args[1]].[k]([args]) end)
  end
end

terra B.Backend:free(library : &opaque) : {}
  self.destroy(self)
  if library ~= nil then 
    OS.FreeLibrary(library)
  end
end

B.Log = terralib.types.funcpointer({&opaque, L.Level, F.conststring}, {}, true)
B.InitBackend = {&opaque, B.Log, M.Behavior} -> &B.Backend

terra LoadDynamicBackend(ui : &opaque, behavior : M.Behavior, log : B.Log, path : rawstring, name : rawstring) : {&B.Backend, &opaque}
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
    name = terralib.select(name == nil, path, name)

    str = Alloc.default_allocator:alloc(int8, C.strlen(name) + 1)
    C.strcpy(str, name)
    name = str

    var ext : rawstring = C.strrchr(str, (".")[0])
    if ext ~= nil then
      @ext = 0
    end
  end

  var f : B.InitBackend = [B.InitBackend](OS.LoadFunction(l, name))

  if str ~= nil then
    Alloc.free(str)
  end

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
terra B.Backend.methods.new(ui : &opaque, behavior : M.Behavior, log : B.Log, path : rawstring, name : rawstring) : {&B.Backend, &opaque}
  return LoadDynamicBackend(ui, behavior, log, path, name)
end

return B