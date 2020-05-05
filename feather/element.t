local F = require 'feather.shared'
local V = require 'feather.virtual'

local struct Elem { }

terra Elem:Construct(target : &opaque) : intptr return -1 end
terra Elem:Destroy() : F.Err return -1 end
terra Elem:Draw(data : &opaque, area : F.Rect) : F.Err return -1 end
terra Elem:MouseDown(x : float, y : float, button : uint8, all : uint8) : F.Err return -1 end
terra Elem:MouseDblClick(x : float, y : float, button : uint8, all : uint8) : F.Err return -1 end
terra Elem:MouseUp(x : float, y : float, button : uint8, all : uint8) : F.Err return -1 end
terra Elem:MouseOn(x : float, y : float, button : uint8, all : uint8) : F.Err return -1 end
terra Elem:MouseOff(x : float, y : float, button : uint8, all : uint8) : F.Err return -1 end
terra Elem:MouseMove(x : float, y : float, button : uint8, all : uint8) : F.Err return -1 end
terra Elem:MouseScroll(x : float, y : float, delta : int16, hdelta : int16) : F.Err return -1 end
terra Elem:TouchBegin(x : float, y : float, index : int16) : F.Err return -1 end
terra Elem:TouchMove(x : float, y : float, index : int16) : F.Err return -1 end
terra Elem:TouchEnd(x : float, y : float, index : int16) : F.Err return -1 end
terra Elem:KeyUp(code : uint8, sigkeys : uint8) : F.Err return -1 end
terra Elem:KeyDown(code : uint8, sigkeys : uint8) : F.Err return -1 end
terra Elem:KeyChar(unicode : int32, sigkeys : uint8) : F.Err return -1 end
terra Elem:JoyButtonDown(button : uint16) : F.Err return -1 end
terra Elem:JoyButtonUp(button : uint16) : F.Err return -1 end
terra Elem:JoyAxis(value : float, axis : uint16) : F.Err return -1 end
terra Elem:SetWindowFlags(flags : uint) : F.Err return -1 end
terra Elem:SetWindowRect(rect : F.Rect) : F.Err return -1 end
terra Elem:GetWindowFlags() : uint return -1 end
terra Elem:Action() : F.Err return -1 end
terra Elem:GotFocus() : F.Err return -1 end
terra Elem:LostFocus() : F.Err return -1 end
terra Elem:DragDrop() : F.Err return -1 end

V.virtualize(Elem)
Elem:complete()

return Elem