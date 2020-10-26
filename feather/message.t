local F = require 'feather.shared'
local Enumset = require 'feather.enumset'
local Virtual = require 'feather.virtual'

local M = {}
M.Idx = uint16

struct M.Receiver { }

terra M.Receiver:Construct(target : &opaque) : intptr return -1 end
terra M.Receiver:Destroy() : F.Err return -1 end
terra M.Receiver:Draw(data : &opaque, area : F.Rect) : F.Err return -1 end
terra M.Receiver:MouseDown(x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return -1 end
terra M.Receiver:MouseDblClick(x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return -1 end
terra M.Receiver:MouseUp(x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return -1 end
terra M.Receiver:MouseOn(x : float, y : float, all : uint8, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:MouseOff(x : float, y : float, all : uint8, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:MouseMove(x : float, y : float, all : uint8, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:MouseScroll(x : float, y : float, delta : float, hdelta : float) : F.Err return -1 end
terra M.Receiver:TouchBegin(x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:TouchMove(x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:TouchEnd(x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:KeyUp(key : uint8, modkeys : uint8, scancode : uint16) : F.Err return -1 end
terra M.Receiver:KeyDown(key : uint8, modkeys : uint8, scancode : uint16) : F.Err return -1 end
terra M.Receiver:KeyChar(unicode : int32, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:JoyButtonDown(index : uint16, button : uint16, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:JoyButtonUp(index : uint16, button : uint16, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:JoyAxis(index : uint16, value : float, axis : uint16, modkeys : uint8) : F.Err return -1 end
terra M.Receiver:JoyOrientation(index : uint16, velocity : F.Vec3D, rotation : F.Vec3D ) : F.Err return -1 end
terra M.Receiver:SetWindowFlags(flags : uint) : F.Err return -1 end
terra M.Receiver:SetWindowRect(rect : F.Rect) : F.Err return -1 end
terra M.Receiver:GetWindowFlags() : uint return -1 end
terra M.Receiver:Action() : F.Err return -1 end
terra M.Receiver:GotFocus() : F.Err return -1 end
terra M.Receiver:LostFocus() : F.Err return -1 end
terra M.Receiver:Drop(kind : int, target : &opaque, count : uint) : F.Err return -1 end

Virtual.virtualize(M.Receiver)
M.Receiver:complete()

struct M.Message {
  kind : M.Idx
  subkind : M.Idx
  union {}
}
M.Message.typelist = {}

struct M.Result {
  union {}
}
M.Result.typelist = {}

local msgenum = {}

-- Go through all our message functions and generate the parameter and result unions
for k, v in pairs(M.Receiver.virtualinfo.info) do
    local f = M.Receiver.methods[k]
    local s = struct{}
    s.name = "Msg"..k
    k = F.camelCase(k)
    s.entries = f.definition.parameters:map(function(e) return {field = e.symbol.displayname, type = e.symbol.type} end)
    table.remove(s.entries, 1) -- Remove self
    M.Message.entries[3]:insert({field = k, type = s})
    M.Message.typelist[v.index] = {field = k, type = s}
    M.Result.entries[1]:insert({field = k, type = f.type.returntype})
    M.Result.typelist[v.index] = f.type.returntype
    table.insert(msgenum, string.upper(k))
    table.insert(msgenum, v.index)
end

M.Kind = Enumset(msgenum, M.Idx)
M.Message.entries[1].type = M.Kind -- Set this to the correct enumeration type

if terralib.sizeof(M.Result) ~= terralib.sizeof(intptr) then
  error("Message.Result ("..terralib.sizeof(M.Result)..") should be the size of a pointer ("..terralib.sizeof(intptr)..")")
end

M.Behavior = {&M.Receiver, &opaque, &opaque, &M.Message} -> M.Result

-- Our default message handler just calls the appropriate vftable pointer and maps our parameters back to it.
terra M.DefaultBehavior(self : &M.Receiver, w : &opaque, ui : &opaque, m : &M.Message) : M.Result
  switch [int](m.kind.val) do
    escape 
      for k, v in pairs(M.Kind.methods) do
        local idx = M.Kind.enum_values[k]
        local args = M.Message.typelist[idx].type.entries:map(function(x) return `m.[M.Message.typelist[idx].field].[x.field] end)
        emit(quote case [idx] then self:virtual([ M.Receiver.virtualinfo.names[idx + 1] ])(self, [args]) end end)
      end
    end
  end
  var r : M.Result;
  r.[M.Message.typelist[0].field] = -1;
  return r
end

return M