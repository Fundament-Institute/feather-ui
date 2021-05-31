local C = terralib.includec "stdio.h"

local M = {}

M.closure = terralib.memoize(
  function(func_type)
    if not terralib.types.istype(func_type) or not (func_type:isfunction() or (func_type:ispointer() and func_type.type:isfunction())) then
      print(func_type)
      print(terralib.types.istype(func_type))
      print(func_type:isfunction())
      print(func_type:ispointer())
      print(func_type.type:isfunction())
      error "type argument must be a function type, or a function pointer type"
    end
    if func_type:ispointer() then
      func_type = func_type.type
    end

    local params = terralib.newlist()
    params:insert(&opaque)
    params:insertall(func_type.parameters)
    
    local struct closure {
      fn: params -> func_type.returntype
      store: &opaque
                         }
    closure.fn_type = params -> func_type.returntype

    local call_params = func_type.parameters:map(symbol)
    terra closure.metamethods.__apply(self: closure, [call_params])
      return self.fn(self.store, [call_params])
    end

    return closure
  end)

return M
