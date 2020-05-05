return function(pairs, type)
  local struct e {
    val: type or uint
  }

  terra e.metamethods.__eq(a : e, b : e) : bool return a.val == b.val end
  terra e.metamethods.__ne(a : e, b : e) : bool return a.val ~= b.val end

  e.enum_values = {}

  local function build(name, value, ...)
    if value ~= nil then
      e.methods[name] = constant(`[e]{[uint](value)})
      e.enum_values[name] = value
      build(...)
    end
  end
  
  e.metamethods.__cast = function(from, to, exp)
    if from == e and to:isintegral() then
      return `[to](exp.val)
    end
    error(("unknown conversion %s to %s"):format(tostring(from),tostring(to)))
  end

  build(unpack(pairs))
  e.convertible = "enum"
  return e
end
