


_color_rgb256 = {}
_color_rgb256['sky'] = (35,192,241)
_color_rgb256['ocean'] = (29,139,204)
_color_rgb256['night'] = (100,120,186)
_color_rgb256['forest'] = (39,182,123)
_color_rgb256['lime'] = (128,189,1)
_color_rgb256['orange'] = (246,147,0)
_color_rgb256['red'] = (246,1,0)

def hexcolor(color):
	c = _color_rgb256[color.casefold()]
	return "#{}{}{}".format(*(hex(c[i])[-2:] if c[i]>15 else "0"+hex(c[i])[-1:] for i in range(3)))

def strcolor_rgb256(color):
	c = _color_rgb256[color.casefold()]
	return "{},{},{}".format(*c)
