local C = terralib.includec "stdio.h"
local cond = require 'std.cond'

local f = require 'feather'
local scaffolding = require 'feather.scaffolding'
local spring = require 'feather.layouts.springgraph'

import 'feather/bind'

local struct springapp {
  nodes: int[3]
  edges: tuple(int,int)[2]
                      }

terra springapp:update()
  
end

terra springapp:init(argc: int, argv: &rawstring)
  for i = 0,3 do
    self.nodes[i] = i
  end
  self.edges[0] = {0,1};
  self.edges[1] = {1,2};
  
  self:update()
end
terra springapp:exit() end

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
  application = &springapp,
  queries = {},
  f.window {
    spring.graph {
      gravity = 1.0,
      drag = 0.1,
      f.each "node" (bind app.nodes) {
        clickable_box {
          pos = spring.node{
            id = bind node,
            mass = 1.0
          },
          ext = bind f.vec3(10, 10, 0),
          onclick = bindevent()
            C.printf("in enablebutton onclick\n")
          end
        }
      }
      --[[f.each "link" (bind app.edges) {
        f.line {
          pos = spring.spring {
            pair = bind link,
            strength = 0.3,
            length = 200.0,
          }
          color = bind fill
        },
      }]]
    }
    ,
  } 
}

return scaffolding.simple_application("springz", springapp, ui, {})
