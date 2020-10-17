local core = require 'feather.core'
local shared = require 'feather.shared'

return core.basic_template {
  color = `[shared.Color]{0}
} (function(context, params)
    return quote
        -- [[
      var x, y = context.transform.r.x, context.transform.r.y
      var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
      var rect = shared.Rect{array(x, y, x+w, y+h)}
      var corners = shared.Rect{array(0f, 0f, 0f, 0f)}
      [context.backend]:DrawRect([context.window], &rect, &corners, [params.color], 0, [params.color], 0, nil, 0, 0, nil)
        --]]
      --[[
    var r = shared.Rect{array(50f, 100f, 200f, 300f)}
    var c = shared.Rect{array(0f, 4f, 8f, 12f)}
    [context.backend]:DrawRect([context.window], &r, &c, 0xFF0000FF, 5f, 0xFF00FFFF, 0f, nil, 0f, 0f, nil)
      --]]
           end
   end
  )
