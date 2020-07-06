local Enumset = require 'feather.enumset'
local Flags = require 'feather.flags'

M = {}
M.Keys = Enumset{
  "NULL", 0, --because its possible to have _lastkey Set to this
  "LBUTTON", 0x01,
  "RBUTTON", 0x02,
  "CANCEL", 0x03,
  "MBUTTON", 0x04,    -- NOT contiguous with L & RBUTTON
  "XBUTTON1", 0x05,    -- NOT contiguous with L & RBUTTON 
  "XBUTTON2", 0x06,    -- NOT contiguous with L & RBUTTON 
  "BACK", 0x08,
  "TAB", 0x09,
  "CLEAR", 0x0C,
  "RETURN", 0x0D,
  "SHIFT", 0x10,
  "CONTROL", 0x11,
  "MENU", 0x12,
  "PAUSE", 0x13,
  "CAPITAL", 0x14,
  "KANA", 0x15,
  "JUNJA", 0x17,
  "FINAL", 0x18,
  "KANJI", 0x19,
  "ESCAPE", 0x1B,
  "CONVERT", 0x1C,
  "NONCONVERT", 0x1D,
  "ACCEPT", 0x1E,
  "MODECHANGE", 0x1F,
  "SPACE", 0x20,
  "PAGEUP", 0x21, -- named PRIOR in windows virtual keys
  "PAGEDOWN", 0x22, -- named NEXT in windows virtual keys
  "END", 0x23,
  "HOME", 0x24,
  "LEFT", 0x25,
  "UP", 0x26,
  "RIGHT", 0x27,
  "DOWN", 0x28,
  "SELECT", 0x29,
  "PRINT", 0x2A,
  "EXECUTE", 0x2B,
  "SNAPSHOT", 0x2C,
  "INSERT", 0x2D,
  "DELETE", 0x2E,
  "HELP", 0x2F,
  "0", 0x30,
  "1", 0x31,
  "2", 0x32,
  "3", 0x33,
  "4", 0x34,
  "5", 0x35,
  "6", 0x36,
  "7", 0x37,
  "8", 0x38,
  "9", 0x39,
  "A", 0x41,
  "B", 0x42,
  "C", 0x43,
  "D", 0x44,
  "E", 0x45,
  "F", 0x46,
  "G", 0x47,
  "H", 0x48,
  "I", 0x49,
  "J", 0x4A,
  "K", 0x4B,
  "L", 0x4C,
  "M", 0x4D,
  "N", 0x4E,
  "O", 0x4F,
  "P", 0x50,
  "Q", 0x51,
  "R", 0x52,
  "S", 0x53,
  "T", 0x54,
  "U", 0x55,
  "V", 0x56,
  "W", 0x57,
  "X", 0x58,
  "Y", 0x59,
  "Z", 0x5A,
  "LWIN", 0x5B,
  "RWIN", 0x5C,
  "APPS", 0x5D,
  "SLEEP", 0x5F,
  "NUMPAD0", 0x60,
  "NUMPAD1", 0x61,
  "NUMPAD2", 0x62,
  "NUMPAD3", 0x63,
  "NUMPAD4", 0x64,
  "NUMPAD5", 0x65,
  "NUMPAD6", 0x66,
  "NUMPAD7", 0x67,
  "NUMPAD8", 0x68,
  "NUMPAD9", 0x69,
  "MULTIPLY", 0x6A,
  "ADD", 0x6B,
  "SEPARATOR", 0x6C,
  "SUBTRACT", 0x6D,
  "DECIMAL", 0x6E,
  "DIVIDE", 0x6F,
  "F1", 0x70,
  "F2", 0x71,
  "F3", 0x72,
  "F4", 0x73,
  "F5", 0x74,
  "F6", 0x75,
  "F7", 0x76,
  "F8", 0x77,
  "F9", 0x78,
  "F10", 0x79,
  "F11", 0x7A,
  "F12", 0x7B,
  "F13", 0x7C,
  "F14", 0x7D,
  "F15", 0x7E,
  "F16", 0x7F,
  "F17", 0x80,
  "F18", 0x81,
  "F19", 0x82,
  "F20", 0x83,
  "F21", 0x84,
  "F22", 0x85,
  "F23", 0x86,
  "F24", 0x87,
  "NUMLOCK", 0x90,
  "SCROLL", 0x91,
  "OEM_NEC_EQUAL", 0x92,   -- '=' key on numpad
  "LSHIFT", 0xA0,
  "RSHIFT", 0xA1,
  "LCONTROL", 0xA2,
  "RCONTROL", 0xA3,
  "LMENU", 0xA4,
  "RMENU", 0xA5,
  "OEM_1", 0xBA,   -- ';:' for US
  "OEM_PLUS", 0xBB,   -- '+' any country
  "OEM_COMMA", 0xBC,   -- ',' any country
  "OEM_MINUS", 0xBD,   -- '-' any country
  "OEM_PERIOD", 0xBE,   -- '.' any country
  "OEM_2", 0xBF,   -- '/?' for US
  "OEM_3", 0xC0,   -- '`virtual ~' for US
  "OEM_4", 0xDB,  --  '[{' for US
  "OEM_5", 0xDC,  --  '\|' for US
  "OEM_6", 0xDD,  --  ']}' for US
  "OEM_7", 0xDE,  --  ''"' for US
  "OEM_8", 0xDF
}

function GenJoyEnum() 
  local l = {}
  for i=0,31 do 
    table.insert(l, "Button"..i+1) 
    table.insert(l, i)
  end 
  for i=0,15 do
    table.insert(l, "ID"..i+1) 
    table.insert(l, i * 256) -- i << 8
  end
  return l
end

M.Joy = Enumset(GenJoyEnum())

M.JoyAxis = Enumset{
  "X", 0,
  "Y", 1,
  "Z", 2,
  "R", 3,
  "U", 4,
  "V", 5,
  "INVALID", 0xFFFF
}

M.MouseButton = Enumset{
  "L", 0x01,
  "R", 0x02,
  "M", 0x10,
  "X1", 0x20,
  "X2", 0x40,
}

M.Window = Flags{
  "MINIMIZABLE",
  "MAXIMIZABLE",
  "RESIZABLE",
  "NOCAPTION", 
  "NOBORDER",
  "MINIMIZED", 
  "MAXIMIZED", 
  "CLOSED",
  "FULLSCREEN",
}

M.ModKey = Flags{
  "SHIFT",
  "CONTROL",
  "ALT",
  "SUPER",
  "CAPSLOCK",
  "NUMLOCK",
  "HELD",
}

M.Touch = Flags{
  "MOVE",
  "HOVER",
  "PALM",
}

return M