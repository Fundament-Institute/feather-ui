

local f = require 'feather'
local scaffolding = require 'feather.scaffolding'

import 'feather/bind'

local struct clickapp {
  state: bool
                      }

terra clickapp:enable()
  self.state = true
end

terra clickapp:disable()
  self.state = false
end

terra clickapp:toggle()
  self.state = not self.state
end

terra clickapp:update()
  --no update behavior needed
end

terra clickapp:init(argc: int, argv: &rawstring)
  self:update()
end
terra clickapp:exit() end

local stroke_color = bind f.Color { 0xccccccff }

local clickable_box = f.template {
  pos = `f.vec3D(0, 0, 0),
  ext = f.required,
  fill = `f.Color {0x000000ff},
  onclick = f.requiredevent()
} {
  f.rect {
    pos = bind pos,
    ext = bind ext,
    outline = stroke_color,
    color = bind fill,
    border = bind 10
  },
  f.mousearea {
    pos = bind pos,
    ext = bind ext,
    onmousedown = bindevent()
      onclick()
    end
  }
}

local ui = f.ui {
  application = &clickapp,
  queries = {},
  f.window {
    clickable_box {
      pos = bind f.vec3D(10, 10, 0),
      ext = bind f.vec3D(100, 100, 0),
      onclick = bindevent()
        app:enable()
      end
    },
    clickable_box {
      pos = bind f.vec3D(120, 10, 0),
      ext = bind f.vec3D(100, 100, 0),
      onclick = bindevent()
        app:toggle()
      end,
      color = bind terralib.select(app.state, f.Color {0xbbbbbbff}, f.Color {0x000000ff})
    },
    clickable_box {
      pos = bind f.vec3D(230, 10, 0),
      ext = bind f.vec3D(100, 100, 0),
      onclick = bindevent()
        app:disable()
      end
    }
  } 
}

return scaffolding.simple_application("clickable", clickapp, ui, {})
