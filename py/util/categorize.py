import re
import numpy



class CategorizerLabel:

	def __init__(self, label, depth=1):
		self.label = label
		self.depth = depth
	
	def setdepth(self, deep):
		self.depth = deep

	def __repr__(self):
		s= "<larch.util.categorize.CategorizerLabel '{}' level {}>".format(self.label,self.depth)
		return s





class Categorizer:

	def __init__(self, label, *members, parent=None):
		if isinstance(label, Categorizer):
			self.label = label.label
			self.members = label.members
			self.depth = label.depth
		else:
			self.label = label
			if len(members)==1 and isinstance(members[0],(list,tuple)):
				self.members = [mem for mem in members[0]]
			else:
				self.members = [mem for mem in members]
			if parent is None:
				self.depth = 1
			elif parent==-1:
				self.depth = 0
			else:
				try:
					self.depth = parent.depth + 1
				except AttributeError:
					self.depth = 1
		for member in members:
			try:
				member.depth = self.depth+1
			except:
				pass

	def complete_members(self):
		x = []
		for member in self.members:
			try:
				x+= list(member.complete_members())
			except AttributeError:
				x+= [member,]
		return x

	def __repr__(self):
		s= "<larch.util.categorize.Categorizer '{}' at {}>".format(self.label,hex(id(self)))
		return s

	def __str__(self):
		s= "Categorizer[{}]".format(self.label)
		for member in self.members:
			s += "\n"
			s += "•"
			s += str(member).replace('\n','\n'+"•")
		return s

	def append(self, member):
		self.members.append(member)
		try:
			self.members[-1].setdepth(self.depth+1)
		except:
			pass

	def extend(self, newmembers):
		self.members += newmembers
		for member in self.members[-len(newmembers):]:
			try:
				member.setdepth(self.depth+1)
			except:
				pass

	def remove(self, thing):
		while thing in self.members: self.members.remove(thing)

	def setdepth(self, deep):
		self.depth = deep
		for member in members:
			try:
				member.setdepth(deep+1)
			except:
				pass

	def match(self, sourcelist, leftovers="Other"):
		sourcelist = numpy.asarray(sourcelist)
		slots = numpy.empty(0, dtype=int)
		subcat = Categorizer(self.label, parent=-1)
		for regex in self.members + ([Categorizer(leftovers, ".*"),] if leftovers else []):
			if isinstance(regex,str):
				newslots = numpy.where([re.search("^{}$".format(regex),x) for x in sourcelist])[0]
				newsubcat = []
			else:
				newslots, newsubcat = regex.match(sourcelist, leftovers=False)
			if len(slots):
				inactive_newslots = numpy.in1d(newslots, slots)
				for dropthis in newslots[inactive_newslots]:
					try:
						newsubcat.remove(sourcelist[dropthis])
					except ValueError:
						pass
				newslots = newslots[~inactive_newslots]
			slots = numpy.append(slots, newslots)
			if isinstance(regex,str):
				subcat.extend([i for i in sourcelist[newslots]])
			else:
				subcat.append(newsubcat)
		return slots, subcat

	def unpack(self):
		first_yield = True
		for member in self.members:
			try:
				for i in member.unpack():
					if first_yield:
						yield CategorizerLabel(self.label, self.depth)
						first_yield = False
					yield i
			except AttributeError:
				if first_yield:
					yield CategorizerLabel(self.label, self.depth)
					first_yield = False
				yield member
		raise StopIteration

class Renamer(Categorizer):

	def decode(self, sourcelist):
		y= self.match(sourcelist, leftovers=False)[0]
		if len(y)==1:
			return sourcelist[y[0]]
		if len(y)>1:
			raise KeyError('multiple matches')
		
	def match(self, sourcelist, leftovers="Other"):
		sourcelist = numpy.asarray(sourcelist)
		slots = numpy.empty(0, dtype=int)
		subcat = Renamer(self.label, parent=-1)
		for regex in self.members:
			if isinstance(regex,str):
				newslots = numpy.where([re.search("^{}$".format(regex),x) for x in sourcelist])[0]
			else:
				newslots, newsubcat = regex.match(sourcelist, leftovers=False)
			if len(slots):
				inactive_newslots = numpy.in1d(newslots, slots)
				newslots = newslots[~inactive_newslots]
			if len(newslots):
				slots = numpy.append(slots, newslots)
				subcat.append(sourcelist[slots[0]])
				return slots, subcat
		return slots, subcat

	def unpack(self):
		if len(self.members):
			yield self
		raise StopIteration

def QuickCategorizer(*categories):
	return Categorizer(None, *(Categorizer(c, "{}.*".format(c)) for c in categories))



if __name__=='__main__':
	c = Categorizer( None,
		Categorizer('Aa', 'A.*'),
		Categorizer('Bb', 'B.*'),
	)


	k = ['A1','A2','B1m','B2t','A3','Cr','B3',]

	z = c.match(k)
	print("\n\nc")
	print(c)

	print('\n\nz[1]')
	print(z[1])

	for j in z[1].unpack():
		print(j)
