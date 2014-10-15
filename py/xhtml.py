

import os
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, TreeBuilder
from contextlib import contextmanager


class Elem(Element):
	def __init__(self, tag, attrib={}, **extra):
		Element.__init__(self,tag,attrib,**extra)
	def put(self, tag, attrib={}, **extra):
		attrib = attrib.copy()
		attrib.update(extra)
		element = Elem(tag, attrib)
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




class XML(TreeBuilder):
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
	def h1(self, content, attrib={}, **extra):
		self.start("h1",attrib, **extra)
		self.data(content)
		self.end("h1")
	def h2(self, content, attrib={}, **extra):
		self.start("h2",attrib, **extra)
		self.data(content)
		self.end("h2")
	def h3(self, content, attrib={}, **extra):
		self.start("h3",attrib, **extra)
		self.data(content)
		self.end("h3")
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



class XHTML(XML):
	def __init__(self, filename=None, *, overwrite=False, spool=True):
		self.root = Elem("html", xmlns="http://www.w3.org/1999/xhtml")
		TreeBuilder.__init__(self, Elem)
		if filename is None:
			import io
			filemaker = lambda: io.BytesIO()
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
		self._f = filemaker()
	def __enter__(self):
		return self
	def __exit__(self, type, value, traceback):
		if type or value or traceback:
			#traceback.print_exception(type, value, traceback)
			return False
		else:
			self.dump()
			self._f.close()
	def dump(self):
		#self._f.write(b'<?xml version="1.0" encoding="UTF-8" ?><!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')
		self._f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')
		xml.etree.ElementTree.ElementTree(self.root).write(self._f, xml_declaration=False, method="html")
		self._f.flush()
		try:
			return self._f.getvalue()
		except AttributeError:
			return
	def append(self, node):
		if isinstance(node, XML):
			self.root.append(node.close())
		elif isinstance(node, Element):
			self.root.append(node)
		else:
			raise TypeError("must be xml.etree.ElementTree.Element or XML or TreeBuilder")

	def __lshift__(self,other):
		self.append(other)
		return self
