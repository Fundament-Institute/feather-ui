local CT = require 'std.constraint'

local function GetBitMask(low, high)
  return bit.lshift((bit.lshift(2, ((high)-(low)))-1), low)
end

local function GetBitType(count)
  if count > 32 then
    return uint64
  end
  return uint32
end

local M = {}
local M_mt = {}
local lookup = {b = 1, g = 2, r = 3, a = 4}

-- Simple integral color representation, designed to mimic GPU color formats which may have uneven bit distributions
M_mt.__call = terralib.memoize(function(self, abits, rbits, gbits, bbits)
  local struct s {
    v : GetBitType(abits + rbits + gbits + bbits)
  }

  s.metamethods.type = s.entries[1].type
  s.metamethods.N = {abits, rbits, gbits, bbits}
  s.metamethods.__typename = function(self) return ("Color:%s[%d:%d:%d:%d]"):format(tostring(self.metamethods.type),self.metamethods.N[1],self.metamethods.N[2],self.metamethods.N[3],self.metamethods.N[4]) end

  local lookuplow = {b = 0, g = bbits, r = gbits + bbits, a = rbits + gbits + bbits }
  local lookuphigh = {b = bbits - 1, g = gbits + bbits - 1, r = rbits + gbits + bbits - 1, a = abits + rbits + gbits + bbits - 1 }
  s.metamethods.__entrymissing = macro(function(entryname, expr)
    if lookuplow[entryname] then
        return `(expr.v and [ GetBitMask(lookuplow[entryname], lookuphigh[entryname]) ]) >> [ lookuplow[entryname] ]
    else
      error (entryname.." is not a valid color component.")
    end
  end)
   
  s.metamethods.__setentry = macro(function(entryname, expr, value)
    if not value:gettype():isintegral() then
      error "tried to set color to non-integral type"
    end
    if lookuplow[entryname] then
      return quote expr.v = (expr.v and not [ GetBitMask(lookuplow[entryname], lookuphigh[entryname]) ]) or ((value << [ lookuplow[entryname] ]) and [ GetBitMask(lookuplow[entryname], lookuphigh[entryname]) ] ) end
    else
      error (entryname.." is not a valid color component.")
    end
  end)

  s.metamethods.__cast = function(from, to, exp)
    if from == s.metamethods.type and to == s then
      return `s{exp}
    end
    error(("unknown conversion %s to %s"):format(tostring(from),tostring(to)))
  end

  return s
end)

-- Generic color representation using half-precision floats, suitable for storing most color formats in most color spaces
--M.H = Ct.Meta({}, 
struct M.H { v : uint16[4] }
M.H.metamethods.__typename = function(self) return "ColorH" end
M.H.metamethods.__entrymissing = macro(function(entryname, expr)
  if lookup[entryname] then
      return `expr.v[ lookup[entryname] ]
  else
    error (entryname.." is not a valid color component.")
  end
end)
M.H.metamethods.__setentry = macro(function(entryname, expr, value)
  if not value:gettype():isarithmetic() then
    error "tried to set color to non-arithmetic type"
  end
  if lookup[entryname] then
    return quote expr.v = value end
  else
    error (entryname.." is not a valid color component.")
  end
end)

return setmetatable(M, M_mt)