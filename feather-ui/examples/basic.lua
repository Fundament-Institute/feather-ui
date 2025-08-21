-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

--local f = require("feather")
local abs, rel, px, NONE = f.abs, f.rel, f.px, f.NONE

local function app(appstate)
	local w = f.window {
		title = "basic-lua",
		resizable = true,
		f.region {
			props = {
				area = abs(90, 90, 0, 200) + rel(0, 0, NONE, 0), -- The NONE in the rect is preserved and turns into a context-dependent value based on what it's assigned to.
				zindex = 0,
			},
			f.button {
				onclick = handlers.onclick, -- This maps to the slot index of this rust function, which will be 0, or slot(APP_SOURCE_ID, 0)
				props = {
					area = abs(45, 45, 0, 0) + rel(0, 0, NONE, 1), -- If the NONE is in an area, it becomes UNSIZED
				},
				f.shape.rect {
					props = {
						area = f.FILL, -- f.FILL is just rel(0, 0, 1, 1)
					},
					corners = 10,
					fill = { 0.2, 0.7, 0.4, 1.0 },
					outline = 0xFFFFFFFF,
				},
				f.text {
					props = {
						area = abs(8, 0, 8, 0) + rel(0, 0.5, NONE, NONE),
						anchor = rel.y(0.5),
					},
					text = "Clicks: " .. appstate.count,
					color = 0xFFFFFFFF,
					fontsize = 40,
					lineheight = 56,
				},
			},
			f.button {
				onclick = handlers.onclick,
				props = {
					area = abs(45, 245, 0, 0) + f.UNSIZED, -- f.UNSIZED is just rel(0, 0, NONE, NONE)
					minsize = abs(100, NONE),
					maxsize = abs(300, NONE),
				},
				f.shape.rect {
					props = {
						area = f.FILL,
					},
					corners = 10, -- shorthand for [10,10,10,10]
					fill = { 0.7, 0.2, 0.4, 1.0 }, -- colors can be arrays of floats
					outline = 0xFFFFFFFF, -- Or specified as an RGBA hex code
				},
				f.text {
					props = {
						area = rel(0.5, 0, NONE, NONE),
						minsize = abs(NONE, 10), -- nil in a limit, however, becomes either NEG_INFINITY for minsize
						maxsize = abs(NONE, 200) + rel(1, NONE), -- or MAX_INFINITY for a maxsize
						anchor = rel.x(0.5),
						padding = abs(8), -- shorthand for abs(8,8,8,8)
					},
					text = string.rep("█", appstate.count),
					color = 0xFFFFFFFF,
					fontsize = 40,
					lineheight = 56,
					wrap = f.WRAP.ANY,
				},
			},
			f.shape.rect {
				props = {
					area = px(1, 1, 2, 2),
				},
				fill = { 1, 1, 1, 1 },
			},
		},
	}

	return appstate, w
end

return app, nil
