

import os
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, TreeBuilder
from contextlib import contextmanager
from ..utilities import uid as _uid
import base64

# @import url(https://fonts.googleapis.com/css?family=Roboto+Mono:400,700,700italic,400italic,100,100italic);


_default_css = """

@import url(https://fonts.googleapis.com/css?family=Roboto:400,700,500italic,100italic|Roboto+Mono:300,400,700);

.error_report {color:red; font-family:monospace;}

body {font-family: "Book Antiqua", "Palatino", serif;}		

table {border-collapse:collapse;}

table, th, td {
	border: 1px solid #999999;
	font-family:"Roboto Mono", monospace;
	font-size:90%;
	font-weight:400;
	}
	
th, td { padding:2px; }

td.parameter_category {
	font-family:"Roboto", monospace;
	font-weight:500;
	background-color: #f4f4f4; 
	font-style: italic;
	}

th {
	font-family:"Roboto", monospace;
	font-weight:700;
	}
	
.larch_signature {font-size:80%; font-weight:100; font-style:italic; }

a.parameter_reference {font-style: italic; text-decoration: none}

.strut2 {min-width:1in}

.histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }

.raw_log pre {
	font-family:"Roboto Mono", monospace;
	font-weight:300;
	font-size:70%;
	}

caption {
    caption-side: bottom;
	text-align: left;
	font-family: Roboto;
	font-style: italic;
	font-weight: 100;
	font-size: 80%;
}
"""







class Elem(Element):
	"""Extends :class:`xml.etree.ElementTree.Element`"""
	def __init__(self, tag, attrib={}, text=None, tail=None, **extra):
		if isinstance(text, Element):
			Element.__init__(self,tag,attrib,**extra)
			for k,v in text.attrib.items():
				if k not in attrib and k not in extra:
					self.set(k,v)
			self.text = text.text
			if tail:
				self.tail = text.tail + tail
			else:
				self.tail = text.tail
		else:
			Element.__init__(self,tag,attrib,**extra)
			if text: self.text = str(text)
			if tail: self.tail = str(tail)
	def put(self, tag, attrib={}, text=None, tail=None, **extra):
		attrib = attrib.copy()
		attrib.update(extra)
		element = Elem(tag, attrib)
		if text: element.text = str(text)
		if tail: element.tail = str(tail)
		self.append(element)
		return element
	def __call__(self, *arg, **attrib):
		for a in arg:
			if isinstance(a,dict):
				for key, value in a.items():
					self.set(str(key), str(value))
			if isinstance(a,str):
				if self.text is None:
					self.text = a
				else:
					self.text += a
		for key, value in attrib.items():
			self.set(str(key), str(value))
		return self
	def append(self, arg):
		if isinstance(arg, XML_Builder):
			super().append(arg.close())   
		else:
			super().append(arg)   
	def __lshift__(self,other):
		if other is not None:
			self.append(other)
		return self
	def tostring(self):
		return xml.etree.ElementTree.tostring(self, encoding="utf8", method="html")

def Anchor_Elem(reftxt, cls, toclevel):
	return Elem('a', {'name':_uid(), 'reftxt':str(reftxt), 'class':str(cls), 'toclevel':str(toclevel)})

def TOC_Elem(reftxt, toclevel):
	return Elem('a', {'name':_uid(), 'reftxt':str(reftxt), 'class':'toc', 'toclevel':str(toclevel)})

class XML_Builder(TreeBuilder):
	"""Extends :class:`xml.etree.ElementTree.TreeBuilder`"""
	def __init__(self, tag=None, attrib={}, **extra):
		TreeBuilder.__init__(self, Elem)
		if tag is None:
			tag="div"
		if tag is not None:
			self.start(tag,attrib,**extra)
	def __getattr__(self, name):
		if len(name)>2 and name[-1]=="_" and name[-2]!="_" and name[0]!="_":
			return self.block(name[:-1])
		if len(name)>4 and name[:4]=="end_":
			i = self.end(name[4:])
			return i
		else:
			i = self.start(name,{})
			return i
	def __call__(self, x):
		return self.data(x)
		
	def title(self, content, attrib={}, **extra):
		self.start("title",attrib, **extra)
		self.data(content)
		self.end("title")
	def anchor(self, ref, reftxt, cls, toclevel):
		self.start("a",{'name':ref, 'reftxt':reftxt, 'class':cls, 'toclevel':toclevel})
		self.end("a")
	def anchor_auto_toc(self, reftxt, toclevel):
		self.start("a",{'name':_uid(), 'reftxt':reftxt, 'class':'toc', 'toclevel':str(toclevel)})
		self.end("a")
	def simple_anchor(self, ref):
		self.start("a",{'name':ref})
		self.end("a")
	def h1(self, content, attrib={}, anchor=None, **extra):
		if anchor:
			self.anchor(_uid(), anchor if isinstance(anchor, str) else content, 'toc', '1')
		self.start("h1",attrib, **extra)
		self.data(content)
		self.end("h1")
	def h2(self, content, attrib={}, anchor=None, **extra):
		if anchor:
			self.anchor(_uid(), anchor if isinstance(anchor, str) else content, 'toc', '2')
		self.start("h2",attrib, **extra)
		self.data(content)
		self.end("h2")
	def h3(self, content, attrib={}, anchor=None, **extra):
		if anchor:
			self.anchor(_uid(), anchor if isinstance(anchor, str) else content, 'toc', '3')
		self.start("h3",attrib, **extra)
		self.data(content)
		self.end("h3")
	def hn(self, n, content, attrib={}, anchor=None, **extra):
		if anchor:
			self.anchor(_uid(), anchor if isinstance(anchor, str) else content, 'toc', '{}'.format(n))
		self.start("h{}".format(n),attrib, **extra)
		self.data(content)
		self.end("h{}".format(n))
	def td(self, content, attrib={}, **extra):
		self.start("td",attrib, **extra)
		self.data(content)
		self.end("td")
	def th(self, content, attrib={}, **extra):
		self.start("th",attrib, **extra)
		self.data(content)
		self.end("th")
	def start(self,tag,attrib={},**extra):
		attrib = attrib.copy()
		attrib.update(extra)
		return TreeBuilder.start(self,tag,attrib)
	def simple(self, tag, content=None, attrib={}, **extra):
		self.start(tag,attrib,**extra)
		if content: self.data(content)
		self.end(tag)
	@contextmanager
	def block(self,tag,attrib={},**extra):
		self.start(tag,attrib,**extra)
		yield
		self.end(tag)

	def dump(self):
		#import io
		#f = io.BytesIO()
		#xml.etree.ElementTree.ElementTree(self.close()).write(f, xml_declaration=False, method="html")
		#return f.getvalue()
		return xml.etree.ElementTree.tostring(self.close())
	def dumps(self):
		return self.dump().decode()

	def append(self, arg):
		div_container = self.start('div')
		if isinstance(arg, XML_Builder):
			div_container.append(arg.close())
		else:
			div_container.append(arg)
		self.end('div')
		
	def __lshift__(self,other):
		self.append(other)
		return self



class XHTML():
	"""A class used to conveniently build xhtml documents."""
	def __init__(self, filename=None, *, overwrite=False, spool=True, quickhead=None, css=None, extra_css=None, view_on_exit=True, jquery=True, jqueryui=True, floating_tablehead=True):
		self.view_on_exit = view_on_exit
		self.root = Elem(tag="html", xmlns="http://www.w3.org/1999/xhtml")
		self.head = Elem(tag="head")
		self.body = Elem(tag="body")
		self.root << self.head
		self.root << self.body
		if filename is None:
			import io
			filemaker = lambda: io.BytesIO()
		elif filename.lower() == "temp":
			from .temporaryfile import TemporaryHtml
			filemaker = lambda: TemporaryHtml(nohead=True)
		else:
			if os.path.exists(filename) and not overwrite and not spool:
				raise IOError("file {0} already exists".format(filename))
			if os.path.exists(filename) and not overwrite and spool:
				filename, filename_ext = os.path.splitext(filename)
				n = 1
				while os.path.exists("{} ({}){}".format(filename,n,filename_ext)):
					n += 1
				filename = "{} ({}){}".format(filename,n,filename_ext)
			filename, filename_ext = os.path.splitext(filename)
			if filename_ext=="":
				filename_ext = ".html"
			filename = filename+filename_ext
			filemaker = lambda: open(filename, 'wb')
		self._filename = filename
		self._f = filemaker()
		self.title = Elem(tag="title")
		self.style = Elem(tag="style")
		from .img import favicon
		self.favicon = Elem(tag="link", attrib={'href':"data:image/png;base64,{}".format(favicon),'rel':"shortcut icon",'type':"image/png"})

		if jquery:
			self.jquery = Elem(tag="script", attrib={
				'src':"https://code.jquery.com/jquery-3.0.0.min.js",
				'integrity':"sha256-JmvOoLtYsmqlsWxa7mDSLMwa6dZ9rrIdtrrVYRnDRH0=",
				'crossorigin':"anonymous",
			})
			self.head << self.jquery

		if jqueryui:
			self.jqueryui = Elem(tag="script", attrib={
				'src':"https://code.jquery.com/ui/1.11.4/jquery-ui.min.js",
				'integrity':"sha256-xNjb53/rY+WmG+4L6tTl9m6PpqknWZvRt0rO1SRnJzw=",
				'crossorigin':"anonymous",
			})
			self.head << self.jqueryui

		if floating_tablehead:
			self.floatThead = Elem(tag="script", attrib={
				'src':"https://cdnjs.cloudflare.com/ajax/libs/floatthead/1.4.0/jquery.floatThead.min.js",
			})
			self.floatTheadA = Elem(tag="script")
			self.floatTheadA.text="""
			$( document ).ready(function() {
				var $table = $('table.floatinghead');
				$table.floatThead({ position: 'absolute' });
				var $tabledf = $('table.dataframe');
				$tabledf.floatThead({ position: 'absolute' });
			});
			$(window).on("hashchange", function () {
				window.scrollTo(window.scrollX, window.scrollY - 50);
			});
			"""
			self.head << self.floatThead
			self.head << self.floatTheadA
		
	

		self.head << self.favicon
		self.head << self.title
		self.head << self.style
		toc_width = 200
		self.toc_color = 'lime'
		default_css = _default_css + """
			
		body { margin-left: """+str(toc_width)+"""px; }
		.table_of_contents_frame { width: """+str(toc_width-13)+"""px; position: fixed; margin-left: -"""+str(toc_width)+"""px; top:0; padding-top:10px; z-index:2000;}
		.table_of_contents { width: """+str(toc_width-13)+"""px; position: fixed; margin-left: -"""+str(toc_width)+"""px; font-size:85%;}
		.table_of_contents_head { font-weight:700; padding-left:25px;  }
		.table_of_contents ul { padding-left:25px;  }
		.table_of_contents ul ul { font-size:75%; padding-left:15px; }
		.larch_signature {font-size:80%; width: """+str(toc_width-30)+"""px; font-weight:100; font-style:italic; position: fixed; left: 0px; bottom: 0px; padding-left:20px; padding-bottom:2px; background-color:rgba(255,255,255,0.9);}
		a.parameter_reference {font-style: italic; text-decoration: none}
		.strut2 {min-width:2in}
		.histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }
		table.floatinghead thead {background-color:#FFF;}
		table.dataframe thead {background-color:#FFF;}
		@media print {
		   body { color: #000; background: #fff; width: 100%; margin: 0; padding: 0;}
		   /*.table_of_contents { display: none; }*/
		   @page {
			  margin: 1in;
		   }
		   h1, h2, h3 { page-break-after: avoid; }
		   img { max-width: 100% !important; }
		   ul, img, table { page-break-inside: avoid; }
		   .larch_signature {font-size:80%; width: 100%; font-weight:100; font-style:italic; padding:0; background-color:#fff; position: fixed; bottom: 0;}
		   .larch_signature img {display:none;}
		   .larch_signature .noprint {display:none;}
		}
		}
		"""

		if quickhead is not None:
			try:
				title = quickhead.title
			except AttributeError:
				title = "Untitled"
			if title != '': self.title.text = str(title)
			if css is None:
				css = default_css
			if extra_css is not None:
				css += extra_css
			try:
				css += quickhead.css
			except AttributeError:
				pass
			self.style.text = css.replace('\n',' ').replace('\t',' ')
			self.head << Elem(tag="meta", name='pymodel', content=base64.standard_b64encode(quickhead.__getstate__()).decode('ascii'))
		else:
			if css is None:
				css = default_css
			if extra_css is not None:
				css += extra_css
			self.style.text = css.replace('\n',' ').replace('\t',' ')


	def __enter__(self):
		return self
	def __exit__(self, type, value, traceback):
		if type or value or traceback:
			#traceback.print_exception(type, value, traceback)
			return False
		else:
			self.dump()
			if self.view_on_exit:
				self.view()
			self._f.close()
			if self._filename is not None and self._filename.lower()!='temp':
				#import webbrowser
				#webbrowser.open('f ile://'+os.path.realpath(self._filename))
				from .temporaryfile import _open_in_chrome_or_something
				_open_in_chrome_or_something('file://'+os.path.realpath(self._filename))

	def toc(self, insert=False):
		xtoc = XML_Builder("div", {'class':'table_of_contents'})
		from .img import local_logo
		logo = local_logo()
		if logo is not None:
			if isinstance(logo,bytes):
				logo = logo.decode()
			xtoc.start('img', attrib={'width':'150','src':"data:image/png;base64,{}".format(logo),
									  'style':'display: block; margin-left: auto; margin-right: auto'})
			xtoc.end('img')
		xtoc.simple('p', content="Table of Contents", attrib={'class':'table_of_contents_head'})
#		for node in self.root.findall('.//a[@toclevel]/..'):
#			anchor = node.find('./a')
#			anchor_name = anchor.get('name')
#			node_text = ""
#			if node.text: node_text += node_text
#			if anchor.text: node_text += anchor.text
#			if anchor.tail: node_text += anchor.tail
#			xtoc.start('li')
#			xtoc.simple('a', content=node_text, attrib={'href':'#{}'.format(anchor_name)})
#			xtoc.end('li')
		toclvl = 0
		for anchor in self.root.findall('.//a[@toclevel]'):
			anchor_ref = anchor.get('name')
			anchor_text = anchor.get('reftxt')
			anchor_lvl = int(anchor.get('toclevel'))
			if anchor_lvl > toclvl:
				xtoc.start('ul')
				toclvl = anchor_lvl
			while anchor_lvl < toclvl:
				xtoc.end('ul')
				toclvl -= 1
			xtoc.start('li')
			xtoc.simple('a', content=anchor_text, attrib={'href':'#{}'.format(anchor_ref)})
			xtoc.end('li')
		#xtoc.end('ul')
		if insert:
			self.body.insert(0,xtoc.close())
		return xtoc.close()

	def toc_iframe(self, insert=False):
		css = """
		.table_of_contents { font-size:85%; font-family:"Book Antiqua", "Palatino", serif; }
		.table_of_contents a:link { text-decoration: none; }
		.table_of_contents a:visited { text-decoration: none; }
		.table_of_contents a:hover { text-decoration: underline; }
		.table_of_contents a:active { text-decoration: underline; }
		.table_of_contents_head { font-weight:700; padding-left:20px }
		.table_of_contents ul { padding-left:20px; }
		.table_of_contents ul ul { font-size:75%; padding-left:15px; }
		::-webkit-scrollbar {
			-webkit-appearance: none;
			width: 7px;
		}
		::-webkit-scrollbar-thumb {
			border-radius: 4px;
			background-color: rgba(0,0,0,.5);
			-webkit-box-shadow: 0 0 1px rgba(255,255,255,.5);
		"""
		xtoc_html = XHTML(css=css)
		xtoc_html.head << Elem(tag='base', target="_parent")
		xtoc_html.body << self.toc()
		
		BLAH = xml.etree.ElementTree.tostring(xtoc_html.root, method="html", encoding="unicode")
		
		
		from .plotting import strcolor_rgb256
		toc_elem = Elem(tag='iframe', attrib={
			'class':'table_of_contents_frame',
			'style':'''height:calc(100% - 100px); border:none; /*background-color:rgba(128,189,1, 0.95);*/
			  background: -webkit-linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* For Safari 5.1 to 6.0 */
			  background: -o-linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* For Opera 11.1 to 12.0 */
			  background: -moz-linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* For Firefox 3.6 to 15 */
			  background: linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* Standard syntax */
			'''.format(strcolor_rgb256(self.toc_color)),
			'srcdoc':BLAH,
			})
		
		if insert:
			self.body.insert(0,toc_elem)
		return toc_elem

	def sign(self, insert=False):
		xsign = XML_Builder("div", {'class':'larch_signature'})
		from .. import longversion as version
		from .img import favicon
		import time
		xsign.start('p')
		xsign.start('img', {'width':"14", 'height':"14", 'src':"data:image/png;base64,{}".format(favicon), 'style':'position:relative;top:2px;' })
		xsign.end('img')
		xsign.data(" Larch {}".format(version))
		xsign.simple('br')
		xsign.data("Report generated on ")
		xsign.simple('br', attrib={'class':'noprint'})
		xsign.data(time.strftime("%A %d %B %Y "))
		xsign.simple('br', attrib={'class':'noprint'})
		xsign.data(time.strftime("%I:%M:%S %p"))
		xsign.end('p')
		if insert:
			self.body.append(xsign.close())
		return xsign.close()
	def dump(self, toc=True, sign=True):
		if sign:
			self.sign(True)
		if toc:
			self.toc_iframe(True)
		self._f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')
		xml.etree.ElementTree.ElementTree(self.root).write(self._f, xml_declaration=False, method="html")
		self._f.flush()
		if sign:
			s = self.root.find(".//div[@class='larch_signature']/..")
			if s is not None:
				s.remove(s.find(".//div[@class='larch_signature']"))
		if toc:
			s = self.root.find(".//div[@class='table_of_contents']/..")
			if s is not None:
				s.remove(s.find(".//div[@class='table_of_contents']"))
		try:
			return self._f.getvalue() # for BytesIO
		except AttributeError:
			return

	def dump_seg(self):
		xml.etree.ElementTree.ElementTree(self.root).write(self._f, xml_declaration=False, method="html")
		self._f.flush()
		try:
			return self._f.getvalue() # for BytesIO
		except AttributeError:
			return

	def view(self):
		try:
			self._f.view()
		except AttributeError:
			pass

	def append(self, node):
		if isinstance(node, XML_Builder):
			self.body.append(node.close())
		elif isinstance(node, Element):
			self.body.append(node)
		elif hasattr(node, '__xml__'):
			self.body.append(node.__xml__())
		else:
			raise TypeError("must be xml.etree.ElementTree.Element or XML_Builder or TreeBuilder or something with __xml__ defined, not {!s}".format(type(node)))

	def __lshift__(self,other):
		self.append(other)
		return self


def xhtml_section_bytes(content):
	import io
	f = io.BytesIO()
	xml.etree.ElementTree.ElementTree(content).write(f, xml_declaration=False, method="html")
	return f.getvalue()

def xhtml_rawtext_as_div(*, filename=None, filehandle=None, classtype='raw_source', title="Source Code"):
	xsource = XML_Builder("div", {'class':classtype})
	use_filehandle = None
	if filename is not None and os.path.isfile(filename):
		use_filehandle = open(filename, 'r')
	if filehandle:
		use_filehandle = filehandle
	if use_filehandle is not None:
		try:
			xsource.h2(title, anchor=1)
			if filename is not None:
				xsource.data("From: {!s}".format(filename))
			xsource.simple("hr")
			xsource.start("pre")
			use_filehandle.seek(0)
			xsource.data(use_filehandle.read())
			xsource.end("pre")
			xsource.simple("hr")
		finally:
			if filename is not None and os.path.isfile(filename):
				use_filehandle.close()
	return xsource.close()

def xhtml_log(f, *, classtype='raw_log', title="Log"):
	if isinstance(f, str):
		filename, filehandle = f, None
	else:
		filename, filehandle = None, f
	return xhtml_rawtext_as_div(filename=filename, filehandle=filehandle, classtype=classtype, title=title)

def xhtml_rawhtml_as_div(contentstring, *, title="And Then", classtype='other_content', headinglevel=2, anchor=1, popper=False):
	xsource = XML_Builder("div", {'class':classtype})
	toggle_id = _uid()
	xsource.start('script')
	xsource.data("""
	$(document).ready(function(){{
		$("#t1{toggle_id}").click(function(){{
			$("#d{toggle_id}").toggle("fast");
		}});
	}});""".format(toggle_id=toggle_id))
	xsource.end('script')
	
	if headinglevel:
		if anchor:
			xsource.anchor(_uid(), anchor if isinstance(anchor, str) else title, 'toc', '{}'.format(headinglevel))
		xsource.start("h{}".format(headinglevel), {'id':'h'+toggle_id})
		xsource.data(title+" ")
		
		if popper:
			from .img import eye
			xsource.start('img',{'id':'t1'+toggle_id, 'src':eye, 'style':'height:24px;vertical-align: text-bottom;'})
			xsource.end('a')
		
		xsource.end("h{}".format(headinglevel))


	if popper:
		div = xsource.start('div', {'id':'d'+toggle_id, 'style':'display:none'})
	else:
		div = xsource.start('div', {'id':'d'+toggle_id})
	if isinstance(contentstring, bytes):
		contentstring = contentstring.decode()
	if isinstance(contentstring, str):
		content = xml.etree.ElementTree.fromstring(contentstring)
	else:
		content = contentstring
	div << content
	xsource.end('div')
	return xsource.close()

def xhtml_dataframe_as_div(contentframe, to_html_kwargs={}, **kwargs):
	return xhtml_rawhtml_as_div(contentframe.to_html(**to_html_kwargs), **kwargs)

def toc_demote_all(elem, demote=1, anchors=True, heads=True):
	for anchor in elem.findall('.//a[@toclevel]'):
		anchor_lvl = int(anchor.get('toclevel'))
		anchor.set('toclevel', str(anchor_lvl+demote))
	for hn in reversed(range(1,6)):
		for h in elem.findall('.//h{}'.format(hn)):
			head_lvl = int(h.tag[1:])
			h.tag = 'h{}'.format(head_lvl+demote)
	return elem




