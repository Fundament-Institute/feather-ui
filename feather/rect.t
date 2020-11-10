local GA = require 'std.ga'
local GA2 = GA(float, 2)
local Math = require 'std.math'

local Rect = terralib.memoize(function(T)
  local struct s {
    ltrb : T[4]
  }

  s.metamethods.type, s.metamethods.N = T, 4
  s.metamethods.__typename = function(self) return ("Rect:%s"):format(tostring(self.metamethods.type)) end
  s.metamethods.__apply = macro(function(self, index) return `self.ltrb[index] end)
  s.methods.Zero = constant(`[s]{array([T](0),[T](0),[T](0),[T](0))})

  local ops = { "__sub","__add","__mul","__div" }
  for _, op in ipairs(ops) do
    local terra array_op(a : T[4], b : T[4])
      var c : s
      for i = 0,4 do
        c.ltrb[i] = operator(op, a[i], b[i])
      end
      return c
    end
    
    local terra SS_op(a : s, b : s) return array_op(a.ltrb, b.ltrb) end
    local terra ST_op(a : s, b : T) return array_op(a.ltrb, array(b, b, b, b)) end
    local terra TS_op(a : T, b : s) return array_op(array(a, a, a, a), b.ltrb) end
    
    s.metamethods[op] = terralib.overloadedfunction("array_op", { SS_op, ST_op, TS_op })
  end

  terra s:Union(x : s) : s
    var b : s
    b.ltrb[0] = Math.min(self.ltrb[0], x.ltrb[0]);
    b.ltrb[1] = Math.min(self.ltrb[1], x.ltrb[1]);
    b.ltrb[2] = Math.max(self.ltrb[2], x.ltrb[2]);
    b.ltrb[3] = Math.max(self.ltrb[3], x.ltrb[3]);
    return b
  end

  terra s:Intersection(x : s) : s
    var b : s
    b.ltrb[0] = Math.max(self.ltrb[0], x.ltrb[0]);
    b.ltrb[1] = Math.max(self.ltrb[1], x.ltrb[1]);
    b.ltrb[2] = Math.min(self.ltrb[2], x.ltrb[2]);
    b.ltrb[3] = Math.min(self.ltrb[3], x.ltrb[3]);
    if b.ltrb[0] > b.ltrb[2] or b.ltrb[1] > b.ltrb[3] then
      return s.Zero
    end
    return b
  end

  terra s.metamethods.__eq(a : s, b : s)
    for i = 0,4 do
      if a.ltrb[i] ~= b.ltrb[i] then return false end
    end
    return true
  end
  
  terra s.metamethods.__ne(a : s, b : s) : bool
    return not [s.metamethods.__eq](a, b)
  end
  
  local lookups = {left = 0, top = 1, right = 2, bottom = 3, l = 0, t = 1, r = 2, b = 3 }
  s.metamethods.__entrymissing = macro(function(entryname, expr)
    if entryname == "topleft" then 
      return `GA2.vector(expr.ltrb[0], expr.ltrb[1])
    end
    if entryname == "bottomright" then 
      return `GA2.vector(expr.ltrb[2], expr.ltrb[3])
    end
    if lookups[entryname] then
      if lookups[entryname] < 4 then
        return `expr.ltrb[ [ lookups[entryname] ] ]
      else
        error ("Index "..lookups[entryname].." doesn't exist in a rect.")
      end
    else
      error (entryname.." is not a valid field.")
    end
  end)
  
  s.metamethods.__setentry = macro(function(entryname, expr, value)
    if value:gettype() ~= t then
      error "tried to set rect element to something other than the rect type"
    end
    if lookups[entryname] then
      if lookups[entryname] < 4 then
        return quote expr.ltrb[ [ lookups[entryname] ] ] = value end
      else
        error ("Index "..lookups[entryname].." doesn't exist in a rect.")
      end
    else
      error (entryname.." is not a valid field.")
    end
  end)

  return s
end)

return Rect