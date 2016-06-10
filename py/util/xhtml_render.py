

import subprocess
import os
import tempfile
import sys


def render_from_htmlfile(filename, pngpath, viewport=(1024,768), clip=True, ext='html'):
	abspath = os.path.abspath(filename)
	
	if os.path.exists(os.path.splitext(pngpath)[0]+'_{}.png'.format(ext)):
		print("png exists:",pngpath)
		return None
	
	print("Rendering:",abspath)
	print("       to:",pngpath)


	js = '''"use strict";
	var page = require('webpage').create();
	'''
	#console.log('The default user agent is ' + page.settings.userAgent);
	#page.settings.userAgent = 'SpecialAgent';

	if viewport is not None:
		# viewportSize being the actual size of the headless browser
		js += 'page.viewportSize = {{ width: {0}, height: {1} }};\n'.format(*viewport)
	
	if viewport is not None and clip:
		# the clipRect is the portion of the page you are taking a screenshot of
		js += 'page.clipRect = {{ top: 0, left: 0, width: {0}, height: {1} }};\n'.format(*viewport)
	
	js += '''
	page.open('file://{}', function (status) {{
		page.evaluate(function() {{
		  document.body.bgColor = 'white';
		}});
		page.render('{}');
		phantom.exit();
	}});'''.format(abspath, pngpath)

	tf = tempfile.NamedTemporaryFile()
	tf.write(js.encode())
	tf.flush()
	tf.seek(0)

	subprocess.call([ 'phantomjs', tf.name ])
	subprocess.call([ 'pngquant', pngpath, '--ext', '_{}.png'.format(ext) ])
	subprocess.call([ 'rm', pngpath ])
	


def render_from_string(s, pngpath, *arg, **kwarg):
	tf = tempfile.NamedTemporaryFile(suffix=".html")
	try:
		tf.write(s.encode())
	except AttributeError:
		tf.write(s)
	tf.flush()
	tf.seek(0)
	render_from_htmlfile(tf.name, pngpath, *arg, **kwarg)



def render_from_docstring_example(obj, pngdir='', **kwarg):
	docstring = obj.__doc__
	
	png_filename = [i.strip()[10:].strip() for i in docstring.split("\n") if i.strip()[:10] == '.. image::'][0]
	png_fileparts = png_filename.split("_")
	png_base = "_".join(png_fileparts[:-1])+".png"
	rendername = os.path.splitext(png_fileparts[-1])[0]
	pngpath = os.path.join(pngdir, png_base)


	print((os.path.splitext(pngpath)[0]+'_{}.png'.format(rendername)))
	if os.path.exists(os.path.splitext(pngpath)[0]+'_{}.png'.format(rendername)):
		print("already exists:", (os.path.splitext(pngpath)[0]+'_{}.png'.format(rendername)), )
		return None

	py = "import larch\n"
	py += "\n".join([i.strip()[4:] for i in docstring.split("\n") if i.strip()[:4] in ('>>> ','... ')])
	pyc = compile(py, "<render_from_docstring_example>", 'exec')
	workspace = {}
	exec(pyc, globals(), workspace)
	
	render_from_string(workspace[rendername], pngpath, ext=rendername, **kwarg)
	print("created:", (os.path.splitext(pngpath)[0]+'_{}.png'.format(rendername)), )


