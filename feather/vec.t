local Vec
Vec = terralib.memoize(function(T, N)
  local struct s {
    v : T[N]
  }

  s.metamethods.type, s.metamethods.N = T, N
  s.metamethods.__typename = function(self) return ("Vec:%s[%d]"):format(tostring(self.metamethods.type),self.metamethods.N) end
  s.metamethods.__apply = macro(function(self, index) return `self.v[index] end)

  local ops = { "__sub","__add","__mul","__div" }
  for _, op in ipairs(ops) do
    local i = symbol(int, "i")
    local function operation(a, b)
      return quote
        var c : s
        for [i] = 0,N do
          c.v[i] = operator(op, a, b)
        end
        return c
      end
    end

    local terra SS_op(a : s, b : s) [operation(`a.v[i], `b.v[i])] end
    local terra ST_op(a : s, b : T) [operation(`a.v[i], `b)] end
    local terra TS_op(a : T, b : s) [operation(`a, `b.v[i])] end
    s.metamethods[op] = terralib.overloadedfunction("vec_op", { SS_op, ST_op, TS_op })
  end

  terra s.metamethods.__eq(a : s, b : s) : bool
    for i = 0,N do
      if a.v[i] ~= b.v[i] then return false end
    end
    return true
  end

  terra s.metamethods.__ne(a : s, b : s) : bool
    return not [s.metamethods.__eq](a, b)
  end

  terra s:norm() : s
    var length = 0
    var normalized : s
    escape
      for i = 0,(N - 1) do
        emit(quote length = length + self.v[i] * self.v[i] end)
      end
      for i = 0,(N - 1) do
        emit(quote normalized.v[i] = self.v[i] / length end)
      end
    end
    return normalized
  end

  local lookups = {x = 0, y = 1, z = 2, w = 3 }
  s.metamethods.__entrymissing = macro(function(entryname, expr)
    if #entryname > 1 then
      return `[Vec(T, #entryname)]{arrayof([T], [terralib.newlist{entryname:byte(1, -1)}:map(string.char):map(function(x) return lookups[x] end)])}
    else
      if lookups[entryname] then
        if lookups[entryname] < N then
          return `expr.v[ [ lookups[entryname] ] ]
        else
          error "Tried to look up letter that doesn't exist in an n-dimensional vector."
        end
      else
        error (entryname.." is not a valid field.")
      end
    end
  end)
   
  s.metamethods.__setentry = macro(function(entryname, expr, value)
    if value:gettype() ~= T then
      error "tried to set vector element to something other than the vector type"
    end
    if lookups[entryname] then
      if lookups[entryname] < N then
        return quote expr.v[ [ lookups[entryname] ] ] = value end
      else
        error "Tried to look up letter that doesn't exist in an n-dimensional vector."
      end
    else
      error (entryname.." is not a valid field.")
    end
  end)
  
  return s
end)

return Vec