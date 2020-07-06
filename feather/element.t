local F = require 'feather.shared'
local V = require 'feather.virtual'

local struct Elem { }

terra Elem:Construct(target : &opaque) : intptr return -1 end
terra Elem:Destroy() : F.Err return -1 end
terra Elem:Draw(data : &opaque, area : F.Rect) : F.Err return -1 end
terra Elem:MouseDown(x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return -1 end
terra Elem:MouseDblClick(x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return -1 end
terra Elem:MouseUp(x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return -1 end
terra Elem:MouseOn(x : float, y : float, all : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:MouseOff(x : float, y : float, all : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:MouseMove(x : float, y : float, all : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:MouseScroll(x : float, y : float, delta : float, hdelta : float) : F.Err return -1 end
terra Elem:TouchBegin(x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:TouchMove(x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:TouchEnd(x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:KeyUp(code : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:KeyDown(code : uint8, modkeys : uint8) : F.Err return -1 end
terra Elem:KeyChar(unicode : int32, modkeys : uint8) : F.Err return -1 end
terra Elem:JoyButtonDown(index : uint16, button : uint16, modkeys : uint8) : F.Err return -1 end
terra Elem:JoyButtonUp(index : uint16, button : uint16, modkeys : uint8) : F.Err return -1 end
terra Elem:JoyAxis(index : uint16, value : float, axis : uint16, modkeys : uint8) : F.Err return -1 end
terra Elem:JoyOrientation(index : uint16, velocity : F.Vec3D, rotation : F.Vec3D ) : F.Err return -1 end
terra Elem:SetWindowFlags(flags : uint) : F.Err return -1 end
terra Elem:SetWindowRect(rect : F.Rect) : F.Err return -1 end
terra Elem:GetWindowFlags() : uint return -1 end
terra Elem:Action() : F.Err return -1 end
terra Elem:GotFocus() : F.Err return -1 end
terra Elem:LostFocus() : F.Err return -1 end
terra Elem:Drop(kind : int, target : &opaque, count : uint) : F.Err return -1 end

V.virtualize(Elem)
Elem:complete()

return Elem