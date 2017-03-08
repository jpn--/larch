

body_font = 'font-family: "Book-Antiqua", "Palatino", serif;'

signature_font = 'font-size:70%; font-weight:100; font-style:italic; font-family: Roboto, Helvetica, sans-serif;'

signature_name_font = 'font-weight:400; font-style:normal; font-family: "Roboto Slab", Roboto, Helvetica, sans-serif;'




def load_css(filename):
	import os
	css = None
	if filename is None or not isinstance(filename, str):
		return None
	if os.path.exists(filename):
		with open(filename, 'r') as f:
			css = f.read()
		return css
	f0 = "{}.css".format(filename)
	if os.path.exists(f0):
		with open(f0, 'r') as f:
			css = f.read()
		return css
	try:
		import appdirs
	except ImportError:
		pass
	else:
		f1 = os.path.join(appdirs.user_config_dir('Larch'), filename)
		if os.path.exists(f1):
			with open(f1, 'r') as f:
				css = f.read()
			return css
		f2 = "{}.css".format(f1)
		if os.path.exists(f2):
			with open(f2, 'r') as f:
				css = f.read()
			return css
	if '{' in filename and '}' in filename:
		return filename
	return css

