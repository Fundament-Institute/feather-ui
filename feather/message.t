local F = require 'feather.shared'
local Enum = require 'feather.enum'

local M = {}
M.Idx = uint16

struct M.Msg {
  type : M.Idx
  subtype : M.Idx
  union {}
}

struct M.Result {
  union {}
}

struct M.Reciever {}

terra M.Reciever:Construct(target : &opaque) : intptr return 0 end
terra M.Reciever:Destroy() : F.Err return 0 end
terra M.Reciever:Draw(data : &opaque, area : F.Rect) : F.Err return 0 end
terra M.Reciever:MouseDown(x : float, y : float, button : uint8, all : uint8) : F.Err return 0 end
terra M.Reciever:MouseDblClick(x : float, y : float, button : uint8, all : uint8) : F.Err return 0 end
terra M.Reciever:MouseUp(x : float, y : float, button : uint8, all : uint8) : F.Err return 0 end
terra M.Reciever:MouseOn(x : float, y : float, button : uint8, all : uint8) : F.Err return 0 end
terra M.Reciever:MouseOff(x : float, y : float, button : uint8, all : uint8) : F.Err return 0 end
terra M.Reciever:MouseMove(x : float, y : float, button : uint8, all : uint8) : F.Err return 0 end
terra M.Reciever:MouseScroll(x : float, y : float, delta : int16, hdelta : int16) : F.Err return 0 end
terra M.Reciever:TouchBegin(x : float, y : float, index : int16) : F.Err return 0 end
terra M.Reciever:TouchMove(x : float, y : float, index : int16) : F.Err return 0 end
terra M.Reciever:TouchEnd(x : float, y : float, index : int16) : F.Err return 0 end
terra M.Reciever:KeyUp(code : uint8, sigkeys : uint8) : F.Err return 0 end
terra M.Reciever:KeyDown(code : uint8, sigkeys : uint8) : F.Err return 0 end
terra M.Reciever:KeyChar(unicode : int32, sigkeys : uint8) : F.Err return 0 end
terra M.Reciever:JoyButtonDown(button : uint16) : F.Err return 0 end
terra M.Reciever:JoyButtonUp(button : uint16) : F.Err return 0 end
terra M.Reciever:JoyAxis(value : float, axis : uint16) : F.Err return 0 end
terra M.Reciever:SetWindowFlags(flags : uint) : F.Err return 0 end
terra M.Reciever:SetWindowRect(rect : F.Rect) : F.Err return 0 end
terra M.Reciever:GetWindowFlags() : uint return 0 end
terra M.Reciever:Action() : F.Err return 0 end
terra M.Reciever:GotFocus() : F.Err return 0 end
terra M.Reciever:LostFocus() : F.Err return 0 end
terra M.Reciever:DragDrop() : F.Err return 0 end

local msgenum = {}

methods = {}
for k in pairs(M.Reciever.methods) do
  table.insert(methods, k)
end
table.sort(methods)

-- Go through all our message functions and generate the parameter and result unions
for i, k in ipairs(methods) do
    local v = M.Reciever.methods[k]
    local s = struct{}
    s.name = "Msg"..k
    k = F.camelCase(k)
    s.entries = v.definition.parameters:map(function(e) return {field = e.symbol.displayname, type = e.symbol.type} end)
    M.Msg.entries[3]:insert({field = k, type = s})
    M.Result.entries[1]:insert({field = k, type = v.type.returntype})
    table.insert(msgenum, string.upper(k))
end

M.Kind = Enum(msgenum, M.Idx)
M.Msg.entries[1].type = M.Kind -- Set this to the correct enumeration type

if terralib.sizeof(M.Result) ~= terralib.sizeof(intptr) then
  error("Msg.Result ("..terralib.sizeof(M.Result)..") should be the size of a pointer ("..terralib.sizeof(intptr)..")")
end

-- This needs to take the base type, and then call MakeBehaviorFunction recursively on that so that only one switch statement
-- pointing to the highest possible function level is created with zero indirection. This may require figuring out what
-- functions have already been added to the switch statement somehow.
function M.MakeBehaviorSwitch(lst, T)

end

-- Alternatively we can simply use the result of __methodmissing to figure out what the function should've resolved to
function M.MakeBehaviorFunction(T) 
  local pairs = {}
  local self = symbol(T)
  for k, v in pairs(T.methods) do
    if M.Reciever.methods[k] ~= nil then
      pairs[msgenum(string.upper(k))] = quote return [self]:[k]() end
    end
  end
end

M.BehaviorFunction = {&opaque, &opaque, &M.Msg} -> M.Result

return M