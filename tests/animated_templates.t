local f = require 'feather'
local scaffolding = require 'feather.scaffolding'
import 'feather/bind'
local shared = require 'feather.shared'

local struct app {}
terra app:init(argc: int, argv: &rawstring) end
terra app:update() end
terra app:exit() end

function gen_stars(num_stars, range_x, range_y)
  local output_table = {}

  math.randomseed(0)

  for i = 0,num_stars do
    local gen_num_x = math.random(range_x[1], range_x[2])
    local gen_num_y = math.random(range_y[1], range_y[2])
    table.insert(output_table, f.circle {
                   color = bind f.Color{0xffffffff},
                   pos = bind f.Vec3{arrayof(float, gen_num_x, gen_num_y, 0)},
                   ext = bind f.Vec3{arrayof(float, 1, 1, 0)}
                   })
  end

  return output_table
end

function export_pieces()
  local output_table = {}
  table.insert(output_table, f.rect {
      color = bind f.Color{0},
      outline = bind f.Color{0xccccccff},
      border = bind 5.0f,
      pos = bind f.Vec3{arrayof(float, 250, 245, 0)},
      corners = bind shared.Rect{array(0f, 0f, 0f, 0f)},
      ext = bind f.Vec3{arrayof(float, 20, 20, 0)}
  })
  table.insert(output_table, f.rect {
      color = bind f.Color{0},
      outline = bind f.Color{0xccccccff},
      border = bind 5.0f,
      pos = bind f.Vec3{arrayof(float, 250, 595, 0)},
      corners = bind shared.Rect{array(0f, 0f, 0f, 0f)},
      ext = bind f.Vec3{arrayof(float, 20, 20, 0)}
  })
  return output_table
end

local ui = f.ui {
  application = &app,
  queries = {},
  f.window {
    color = bind f.Color{0},
    size = bind f.Vec{array(500f, 700f)}, -- x, y
    resizable = bind true,
    windowname = bind "DON'T PANIC",

    unpack(export_pieces()),
    unpack(gen_stars(150, {10, 490}, {10, 490}))
  }
}

return scaffolding.simple_application("rectangle", app, ui, {})
