local tunpack = unpack or table.unpack
local Math = require 'std.math'
local M = {}

function M.astype(typ)
  if terralib.types.istype(typ) then
    return typ
  elseif terralib.isquote(typ) then
    return typ:astype()
  else
    error("tried to convert something to a type that isn't a quote or a type")
  end
end

function M.type_template(typefn)
  typefn = terralib.memoize(typefn)
  local function fn(...)
    local args = {...}
    for i = 1, select("#", ...) do
      args[i] = M.astype(args[i])
    end
    return typefn(tunpack(args))
  end
  return macro(fn, fn)
end

function M.SafeFree(...)
  return quote
    escape
      for _, p in ipairs({...}) do
        if not p:gettype():ispointer() then
          error ("p is not a pointer: " .. p:gettype():layoutstring())
        end
        emit(quote 
          if p ~= nil then
            C.free(p)
          end
        end)
      end
    end
  end
end

local terra unalignedload(addr : &float) : vector(float,4)
	return terralib.attrload([&vector(float,4)](addr), { align = 4 })
end
local terra unalignedstore(addr : &float, value : vector(float,4))
	terralib.attrstore([&vector(float,4)](addr),value, { align = 4 })
end

local function genmm(v, i, j, m)
  return `v[j] * vector(m[i][j], m[i][j], m[i][j], m[i][j])
end

terra M.MatrixMultiply4x4(l : float[4][4], r : float[4][4], out : &float)
  var v : vector(float,4)[4]
  for k = 0, 4 do 
    v[k] = unalignedload(r[k])
  end

  escape
    for i = 0, 3 do 
      emit(quote unalignedstore(out + [i*4], [genmm(v, i, 0, l)] + [genmm(v, i, 1, l)] + [genmm(v, i, 2, l)] + [genmm(v, i, 3, l)]) end)
    end
  end
end

local function genvm(r, j, m)
  return `unalignedload(r[j]) * vector(m[j], m[j], m[j], m[j])
end

terra M.VecMultiply3D(l : float[3], r : float[4][4]) : float[3]
  var out : float[4]
  unalignedstore(out, [genvm(r, 0, l)] + [genvm(r, 1, l)] + [genvm(r, 2, l)])
  return array(out[0], out[1], out[2])
end

M.MatrixIdentity = constant(`array(array(1.0f,0.0f,0.0f,0.0f),array(0.0f,1.0f,0.0f,0.0f),array(0.0f,0.0f,1.0f,0.0f),array(0.0f,0.0f,0.0f,1.0f)))

terra M.MatrixRotation(a : float, b : float, c : float) : float[4][4]
  var sina = Math.sin(a)
  var sinb = Math.sin(b)
  var sinc = Math.sin(c)
  var cosa = Math.cos(a)
  var cosb = Math.cos(b)
  var cosc = Math.cos(c)
  return array(
    array(cosa*cosb, cosa*sinb*sinc - sina*cosc, cosa*sinb*cosc + sina*sinc, 0.0f),
    array(sina*cosb, sina*sinb*sinc + cosa*cosc, sina*sinb*cosc - cosa*sinc, 0.0f),
    array(-sinb    , cosb*sinc                 , cosb*cosc                 , 0.0f),
    array(0.0f     , 0.0f                      , 0.0f                      , 1.0f)
  )
end

terra M.Rotate2D(x : float, y : float, r : float, outx : &float, outy : &float) : {}
  var cosr = Math.cos(r)
  var sinr = Math.cos(r)
  @outx = x * cosr - y * sinr
  @outy = y * cosr + x * sinr
end

terra M.MatrixAbs(m : &float[4][4])
  escape
    for i = 0, 3 do 
      for j = 0, 3 do 
        emit(quote (@m)[i][j] = Math.abs((@m)[i][j]) end)
      end
    end
  end
end


function M.override(a, b)
  local res = {}
  for k, v in pairs(a) do res[k] = v end
  for k, v in pairs(b) do res[k] = v end
  return res
end

return M
