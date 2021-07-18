
local M = {}

local virtual_invoke = macro(function(obj, funcname)
  local result = macro(function(self, ...)
      local args = {...}
      local info = obj:gettype().virtualinfo.info[funcname:asvalue()]
      if not info then
        error("type "..tostring(obj:gettype()) .. " does not have a virtual method named "..funcname:asvalue())
        end
      --TODO: handle overloaded functions
      return quote
          var temp = [ not obj:islvalue() and not obj:gettype():ispointer() and obj or `&obj ]
        in
          [info.pointertype](temp.vftable[ [info.index] ])(
            [info.selftype]([temp.type:ispointer() and temp or `&temp]),
            [args]
                                                          )
             end
  end)
  return result
end)

local function get_funcpointer(class, name)
  if class.methods[name] and terralib.isfunction(class.methods[name]) then
    return `[&opaque]([class.methods[name] ])
  elseif class.parent then
    return get_funcpointer(class.parent, name)
  else
    error("unable to find a method implementation for "..name)
  end
end

local function virtual_staticinitialize(class)
  local info = {}
  local names = terralib.newlist()
  for k, v in pairs(class.methods) do
    if terralib.isfunction(v) then
      --TODO: handle overloaded functions
      if v:gettype().parameters[1] == &class then
        info[k] = {
          selftype = &class,
          pointertype = &(v:gettype())
        }
        if not class.parent or not class.parent.virtualinfo.info[k] then
          names:insert(k)
        end
      end
    end
  end
  names:sort()
  for i, name in ipairs(names) do
    info[name].index = i - 1 + (class.parent and class.parent.virtualinfo.count or 0)
  end
  local combined_names
  if class.parent then
    for k, v in pairs(class.parent.virtualinfo.info) do
      info[k] = v
    end
    combined_names = terralib.newlist()
    for i, v in ipairs(class.parent.virtualinfo.names) do
      combined_names[i] = v
    end
    for i, v in ipairs(names) do
      combined_names:insert(v)
    end
  else
    combined_names = names
  end
  class.methods.virtual_initializer = global(`arrayof([&opaque], [combined_names:map(function(name) return get_funcpointer(class, name) end)]))
  class.virtualinfo = {
    info = info,
    names = combined_names,
    count = #combined_names,
  }
  class.methods.virtual = virtual_invoke
end

-- You must MANUALLY initialize vftable by assigning it to .virtual_initializer
function M.virtualize(class)
  table.insert(class.entries, 1, {field = "vftable", type = &&opaque})
  class.metamethods.__staticinitialize = virtual_staticinitialize
end

function M.extends(parent)
  return function(class)
    table.insert(class.entries, 1, {field = "super", type = parent})
    for _, name in ipairs(parent.virtualinfo.names) do
      if not class.methods[name] then
        class.methods[name] = macro(function(self, ...)
          local args = {...}
            return `self.super:[name]([args])
        end)
      end
    end
    class.parent = parent
    class.metamethods.__staticinitialize = virtual_staticinitialize
    class.metamethods.__entrymissing = macro(function(entryname,myobj)
      return `myobj.super.[entryname]
    end)
  end
end

return M
