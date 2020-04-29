local C = terralib.includecstring [[
  #include <string.h>
]]

local B = {}
local F = require 'feather.shared'
local Flags = require 'feather.flags'
local Enum = require 'feather.enum'
local OS = require 'feather.os'
local Alloc = require 'std.alloc'
local M = require 'feather.message'
local L = require 'feather.log'

B.Features = Flags{
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

B.Formats = Enum{
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

B.Formats.methods["UNKNOWN"] = constant(`0xff)
B.Formats.enum_values["UNKNOWN"] = 0xff

B.AntiAliasing = Enum{
  "NO_AA",
  "AA",
  "LCD",
  "LCD_V"}

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
}

struct B.Asset {
  data : B.Data
  size : F.Veci
  dpi : F.Vec
  format : B.Formats
}

struct B.Display {
  size : F.Veci
  offset : F.Veci
  dpi : F.Vec
  scale : float
  handle : uint64
  primary : bool
}

struct B.Backend {
  destroy : {&B.Backend} -> {}

  features : B.Features
  formats : uint
  dpi : F.Vec
  scale : float
  cursorblink : uint64
  tooltipdelay : uint64
}

terra B.Backend:DrawFont(data : &opaque, font : &B.Font, layout : &opaque, area : &F.Rect, color : F.Color, lineHeight : float, letterSpacing : float, blur : float, aa : B.AntiAliasing) : F.Err return 0 end
terra B.Backend:DrawAsset(data : &opaque, asset : &B.Asset, area : &F.Rect, source : &F.Rect, color : F.Color, time : float) : F.Err return 0 end
terra B.Backend:DrawRect(data : &opaque, area : &F.Rect, corners : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset) : F.Err return 0 end
terra B.Backend:DrawCircle(data : &opaque, area : &F.Rect, arcs : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset) : F.Err return 0 end
terra B.Backend:DrawTriangle(data : &opaque, area : &F.Rect, corners : &F.Rect, fillColor : F.Color, border : float, borderColor : F.Color, blur : float, asset : &B.Asset) : F.Err return 0 end
terra B.Backend:DrawLines(data : &opaque, points : &F.Vec, count : uint, color : F.Color) : F.Err return 0 end
terra B.Backend:DrawCurve(data : &opaque, anchors : &F.Vec, count : uint, fillColor : F.Color, stroke : float, strokeColor : F.Color) : F.Err return 0 end
terra B.Backend:PushLayer(data : &opaque, area : F.Rect, transform : &float, opacity : float) : F.Err return 0 end
terra B.Backend:PopLayer(data : &opaque) : F.Err return 0 end
terra B.Backend:PushClip(data : &opaque, area : F.Rect) : F.Err return 0 end
terra B.Backend:PopClip(data : &opaque) : F.Err return 0 end
terra B.Backend:DirtyRect(data : &opaque, area : F.Rect) : F.Err return 0 end

terra B.Backend:CreateFont(family : F.conststring, weight : uint16, italic : bool, pt : uint32, dpi : F.Vec) : &B.Font return nil end
terra B.Backend:DestroyFont(font : &B.Font) : F.Err return 0 end
terra B.Backend:FontLayout(font : &B.Font, text : F.conststring, area : &F.Rect, lineHeight : float, letterSpacing : float, previous : &opaque, dpi : F.Vec) : &opaque return nil end
terra B.Backend:FontIndex(font : &B.Font, layout : &opaque, area : &F.Rect, lineHeight : float, letterSpacing : float, pos : F.Vec, cursor : &F.Vec, dpi : F.Vec) : uint return 0 end
terra B.Backend:FontPos(font : &B.Font, layout : &opaque, area :&F.Rect, lineHeight : float, letterSpacing : float, index : uint, dpi : F.Vec) : F.Vec return F.Vec{} end

terra B.Backend:CreateAsset(data : F.conststring, count : uint, format : B.Formats) : &B.Asset return nil end
terra B.Backend:DestroyAsset(asset : &B.Asset) : F.Err return 0 end

terra B.Backend:PutClipboard(kind : B.Clipboard, data : F.conststring, count : uint) : F.Err return 0 end
terra B.Backend:GetClipboard(kind : B.Clipboard, target : &opaque, count : uint) : uint return 0 end
terra B.Backend:CheckClipboard(kind : B.Clipboard) : bool return false end
terra B.Backend:ClearClipboard(kind : B.Clipboard) : F.Err return 0 end

terra B.Backend:GetSyncObject() : &opaque return nil end
terra B.Backend:ProcessMessages() : F.Err return 0 end
terra B.Backend:SetCursor(window : &opaque, cursor : B.Cursor) : F.Err return 0 end
terra B.Backend:RequestAnimationFrame(window : &opaque, delay : uint64) : F.Err return 0 end
terra B.Backend:GetDisplayIndex(index : uint, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplay(handle : uint64, out : &B.Display) : F.Err return 0 end
terra B.Backend:GetDisplayWindow(window : &opaque, out : &B.Display) : F.Err return 0 end
terra B.Backend:CreateWindow(data : &opaque, display : uint64, area : &F.Rect, caption : F.conststring, flags : uint64) : &opaque return nil end
terra B.Backend:SetWindow(window : &opaque, data : &opaque, display : uint64, area : &F.Rect, caption : F.conststring, flags : uint64) : F.Err return 0 end
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

B.Log = terralib.types.funcpointer({&opaque, L.Level, F.conststring}, &B.Backend, true)
B.Behavior = {&opaque, &opaque, &M.Msg} -> M.Result
B.InitBackend = {&opaque, B.Log, B.Behavior} -> &B.Backend

terra LoadDynamicBackend(ui : &opaque, behavior : B.Behavior, log : B.Log, path : rawstring, name : rawstring) : {&B.Backend, &opaque}
  var l : &opaque = OS.LoadLibrary(path)

  if l == nil then
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
  end

  OS.FreeLibrary(l)
  return nil, nil
end

-- In cases where the backend is not known at compile time, we must load it via a shared library at runtime 
function B.Backend:new(root, path, name)
  return quote in LoadDynamicBackend(root, path, name) end
end

return B