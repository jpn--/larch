

import os
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, TreeBuilder
from contextlib import contextmanager
from ..utilities import uid as _uid
import base64

class Elem(Element):
	"""Extends :class:`xml.etree.ElementTree.Element`"""
	def __init__(self, tag, attrib={}, text=None, **extra):
		Element.__init__(self,tag,attrib,**extra)
		if text: self.text = text
	def put(self, tag, attrib={}, text=None, **extra):
		attrib = attrib.copy()
		attrib.update(extra)
		element = Elem(tag, attrib)
		if text: element.text = text
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
		self.append(other)
		return self

def Anchor_Elem(reftxt, cls, toclevel):
	return Elem('a', {'name':_uid(), 'reftxt':str(reftxt), 'class':str(cls), 'toclevel':str(toclevel)})

def TOC_Elem(reftxt, toclevel):
	return Elem('a', {'name':_uid(), 'reftxt':str(reftxt), 'class':'toc', 'toclevel':str(toclevel)})

class XML_Builder(TreeBuilder):
	"""Extends :class:`xml.etree.ElementTree.TreeBuilder`"""
	def __init__(self, tag=None, attrib={}, **extra):
		if tag is None: tag="div"
		TreeBuilder.__init__(self, Elem)
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



class XHTML():
	"""A class used to conveniently build xhtml documents."""
	def __init__(self, filename=None, *, overwrite=False, spool=True, quickhead=None, css=None, view_on_exit=True):
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
		self.head << self.title
		self.head << self.style
		default_css = """
		.error_report {color:red; font-family:monospace;}
		table {border-collapse:collapse;}
		table, th, td {border: 1px solid #999999; padding:2px; font-family:monospace;}
		body { margin-left: 200px; }
		.table_of_contents_frame { width: 190px; position: fixed; margin-left: -200px; top:0; padding-top:10px;}
		.table_of_contents { width: 190px; position: fixed; margin-left: -200px; font-size:85%; }
		.table_of_contents_head { font-weight:700; padding-left:25px }
		.table_of_contents ul { padding-left:25px; }
		.table_of_contents ul ul { font-size:75%; padding-left:15px; }
		.larch_signature {font-size:80%; width: 170px; font-weight:100; font-style:italic; position: fixed; left: 0px; bottom: 0px; padding-left:20px; padding-bottom:2px; background-color:rgba(255,255,255,0.9);}
		a.parameter_reference {font-style: italic; text-decoration: none}
		.strut2 {min-width:2in}
		"""

		if quickhead is not None:
			try:
				title = quickhead.title
			except AttributeError:
				title = "Untitled"
			if title != '': self.title.text = str(title)
			if css is None: css = default_css
			try:
				css += quickhead.css
			except AttributeError:
				pass
			self.style.text = css.replace('\n',' ').replace('\t',' ')
			self.head << Elem(tag="meta", name='pymodel', content=base64.standard_b64encode(quickhead.__getstate__()).decode('ascii'))
		else:
			if css is None: css = default_css
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
		.table_of_contents { font-size:85%; }
		.table_of_contents_head { font-weight:700; padding-left:25px }
		.table_of_contents ul { padding-left:25px; }
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
		
		toc_elem = Elem(tag='iframe', attrib={
			'class':'table_of_contents_frame',
			'style':'''height:calc(100% - 100px); border:none; /*background-color:rgba(255, 255, 200, 0.9);*/
			  background: -webkit-linear-gradient(rgba(255, 255, 200, 0.9), rgba(255, 255, 255, 0.9)); /* For Safari 5.1 to 6.0 */
			  background: -o-linear-gradient(rgba(255, 255, 200, 0.9), rgba(255, 255, 255, 0.9)); /* For Opera 11.1 to 12.0 */
			  background: -moz-linear-gradient(rgba(255, 255, 200, 0.9), rgba(255, 255, 255, 0.9)); /* For Firefox 3.6 to 15 */
			  background: linear-gradient(rgba(255, 255, 200, 0.9), rgba(255, 255, 255, 0.9)); /* Standard syntax */
			''',
			'srcdoc':BLAH,
			})
		
		if insert:
			self.body.insert(0,toc_elem)
		return toc_elem

	def sign(self, insert=False):
		xsign = XML_Builder("div", {'class':'larch_signature'})
		from ..version import version
		import time
		xsign.start('p')
		xsign.data("Larch {}".format(version))
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
		else:
			raise TypeError("must be xml.etree.ElementTree.Element or XML_Builder or TreeBuilder")

	def __lshift__(self,other):
		self.append(other)
		return self




def toc_demote_all(elem, demote=1, anchors=True, heads=True):
	for anchor in elem.findall('.//a[@toclevel]'):
		anchor_lvl = int(anchor.get('toclevel'))
		anchor.set('toclevel', str(anchor_lvl+demote))
	for hn in reversed(range(1,6)):
		for h in elem.findall('.//h{}'.format(hn)):
			head_lvl = int(h.tag[1:])
			h.tag = 'h{}'.format(head_lvl+demote)
	return elem



