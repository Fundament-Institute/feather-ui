

local M = {}


function M.whereis(fn, name)
  local info = debug.getinfo(fn, "S")
  print("function ", name, info.source..":"..info.linedefined)
  return fn
end

return M
