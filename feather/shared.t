local S = {}

S.Err = int
S.Delegate = &opaque -> {}
S.Version = { 0, 1, 0 }

local Color = require 'feather.color'

S.Color = Color(8,8,8,8)
S.Color16 = Color(16,16,16,16)

S.Color.c_export = [[union {
    unsigned int v;
    unsigned char colors[4];
    struct
    {
      unsigned char b;
      unsigned char g;
      unsigned char r;
      unsigned char a;
    };
  };]]
local Rect = require 'feather.rect'

S.Rect = Rect(float)
S.Rect.c_export = [[union {
    struct
    {
      float left;
      float top;
      float right;
      float bottom;
    };

    float ltrb[4];
  };]]

struct S.URect {
  abs : S.Rect 
  rel : S.Rect
}

local GA = require 'std.ga'
local Iter = require 'std.iterator'

local veclookups = { [1] = "x", [2] = "y", [4] = "z", [8] = "w" }

local function VecExport(v)
  local type = v.entries[1].type.type == int and "int" or tostring(v.entries[1].type.type)
  v.c_export = ""
  for x,i in ipairs(v.metamethods.components) do
    local name = type .. " "
    for k,c in pairs(veclookups) do
      if bit.band(k, i) ~= 0 then
        name = name .. c
      end
    end
    v.c_export = v.c_export..name..";\n    "
  end
end

function AutoVecFunc(type)
  return macro(function(...) 
    local args = {...}
    return `[type] { arrayof([type.entries[1].type.type], [args]) }
  end)
end

local prefixes = { "Vec", "Bivector", "Trivector", "Quadvector" }

function AutoVecGen(type, min, max)
  local tpost = type == float and "" or "i"
  for d=min, max do
    local post = d == 2 and tpost or tpost..d
    for i=1,d do 
      local v = prefixes[i] .. post
      S[v] = GA(type, d)["vector"..i.."_t"]
      VecExport(S[v])
      local k = string.lower(v:sub(1,1)) .. v:sub(2, -1)
      S[k] = AutoVecFunc(S[v])
    end
  end
end

AutoVecGen(float, 2, 4)
AutoVecGen(int, 2, 2)

S.rect = AutoVecFunc(S.Rect)
S.scalar_t = GA(float, 3).scalar_t

struct S.UVec {
  abs : S.Vec
  rel : S.Vec
}

struct S.conststring {
  s : rawstring
}
S.conststring.metamethods.__cast = function(from, to, exp)
  if from == rawstring and to == S.conststring then
    return `S.conststring{exp}
  end
  if from == niltype and to == S.conststring then
    return `S.conststring{nil}
  end
  if from == S.conststring and to == rawstring then
    return `exp.s
  end
  if from == niltype and to == rawstring then
    return `nil
  end
  error(("unknown conversion %s to %s"):format(tostring(from),tostring(to)))
end

function S.camelCase(k) return string.lower(k:sub(1,1)) .. k:sub(2, -1) end

-- Accepts any number of arguments and always returns -1
struct S.EmptyCallable {}
S.EmptyCallable.metamethods.__apply = macro(function(...) return `-1 end)

return S
