local function Flags(lst)
  local map = {}
  for i, v in ipairs(lst) do
    map[v] = i - 1
  end

  local struct flagset {
    val: uint
  }

  flagset.metamethods.__add = macro(function(a, b)
      return `[flagset]{a.val or b.val}
  end)

  flagset.metamethods.__entrymissing = macro(function(entryname, flag_expr)
      if map[entryname] then
        return `(flag_expr.val and [2^map[entryname]]) ~= 0
      else
        error (entryname.." is not a flag in this set")
      end
  end)

  flagset.metamethods.__setentry = macro(function(entryname, flag_expr, value_expr)
      if value_expr:gettype() ~= bool then
        error "Flags can only be true or false."
      end
      if map[entryname] then
        return quote flag_expr.val = (flag_expr.val and not [2^map[entryname]]) or terralib.select(value_expr, [2^map[entryname]], 0) end
      else
        error (entryname.." is not a flag in this set")
      end
  end)

  flagset.metamethods.__cast = function(from, to, exp)
    if from == flagset and to:isintegral() then
      return `[to](exp.val)
    end
    error(("unknown conversion %s to %s"):format(tostring(from),tostring(to)))
  end

  flagset.enum_values = {}

  for i, v in ipairs(lst) do
    flagset.methods[v] = constant(`[flagset]{[2^(i - 1)]})
    flagset.enum_values[v] = 2^(i - 1)
  end

  flagset.convertible = "enum"

  return flagset
end

return Flags
