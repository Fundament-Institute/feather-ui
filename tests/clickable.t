local C = terralib.includec "stdio.h"
local cond = require 'std.cond'

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

local stroke_color = bind f.Color { 0xffcccccc }

local clickable_box = f.template {
  pos = `f.vec3(0, 0, 0),
  ext = f.required,
  fill = `f.Color {0xff000000},
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
    mousedown = bindevent()
      C.printf("in clickable_box mousedown\n")
      onclick()
      return 1
    end
  }
}

local ui = f.ui {
  application = &clickapp,
  queries = {},
  f.window {
    clickable_box {
      pos = bind f.vec3(60, 60, 0),
      ext = bind f.vec3(50, 50, 0),
      onclick = bindevent()
        C.printf("in enablebutton onclick\n")
        app:enable()
      end
    },
    clickable_box {
      pos = bind f.vec3(170, 60, 0),
      ext = bind f.vec3(50, 50, 0),
      onclick = bindevent()
        C.printf("in togglebutton onclick\n")
        app:toggle()
      end,
      fill = bind terralib.select(app.state, f.Color {0xffbbbbbb}, f.Color {0xff000000})
    },
    clickable_box {
      pos = bind f.vec3(280, 60, 0),
      ext = bind f.vec3(50, 50, 0),
      onclick = bindevent()
        C.printf("in disablebutton onclick\n")
        app:disable()
      end
    }
  } 
}

return scaffolding.simple_application("clickable", clickapp, ui, {})
