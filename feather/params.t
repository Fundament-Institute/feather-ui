local F = require 'feather.shared'
local Expression = require 'feather.expression'

local M = {}
M.required = {}
M.outline_mt = {__index = {}, __call = function(self, ...) return self:generate(...) end}
M.typed_event_spec_mt = {}
M.event_spec_mt = {
    __call = function(self, ...)
      return setmetatable({args = {...}, required = self.required}, M.typed_event_spec_mt)
    end
}
-- body is used both as a distinguished unique value and as an outline which looks up the provided body in the enclosing environment under it's own key
M.body = setmetatable({
    generate = function(type_environment)
      return type_environment[M.body](type_environment)
    end
    }, M.outline_mt)

M.event = setmetatable({required = false}, M.event_spec_mt)
M.requiredevent = setmetatable({required = true}, M.event_spec_mt)

-- parse an argument declaration table into a usable specification object
function M.parse_params(desc)
  local res = {required = {}, defaults = {}, names = {}, names_map = {}, events = {}, has_body = false}
  for k, v in pairs(desc) do
    if type(k) ~= "string" then
      if k == 1 and v == M.body then
        res.has_body = true
      else
        error("invalid argument specifier provided to declaration", 2)
      end
    else
      table.insert(res.names, k)
      res.names_map[k] = true
      if v == M.required then
        res.required[k] = true
      else
        local val_mt = getmetatable(v)
        if val_mt == M.event_spec_mt or val_mt == M.typed_event_spec_mt then
          if v.required then
            res.required[k] = true
          else
            res.defaults[k] = Expression.constant(`F.EmptyCallable {})
          end
          res.events[k] = v
        else
          res.defaults[k] = Expression.constant(v)
        end
      end
    end
  end
  return res
end

return M
