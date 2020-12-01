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

    struct
    {
      FG_Vec topleft;
      FG_Vec bottomright;
    };

    float ltrb[4];
  };]]

struct S.URect {
  abs : S.Rect 
  rel : S.Rect
}

local GA = require 'std.ga'
local GA3 = GA(float, 3)

S.Vec = GA(float, 2).vector_t
S.Vec3D = GA3.vector_t
S.Bivector3 = GA3.bivector_t
S.Veci = GA(int, 2).vector_t

local veclookups = { [1] = "x", [2] = "y", [4] = "z", [8] = "w" }

local function vecexport(v, type)
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

vecexport(S.Vec, "float")
vecexport(S.Vec3D, "float")
vecexport(S.Bivector3, "float")
vecexport(S.Veci, "int")

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
