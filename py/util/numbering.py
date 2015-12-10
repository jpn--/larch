from enum import Enum

def enum_bitmask(enumeration):
	mask = 0
	for i in enumeration:
		mask |= i.value
	return mask

def enum_bitmask_length(enumeration):
	return enum_bitmask(enumeration).bit_length()

class numbering_system():
	def __init__(self, *enumerations):
		self.enumerations = enumerations
		self.bitmasks = [ enum_bitmask(i) for i in enumerations ]
		self.bitmask_lengths = [ enum_bitmask_length(i) for i in enumerations ]
		self.shifts = [ ]
		self.total_shift = 0
		for j in range(len(self.enumerations)):
			shift = self.bitmask_lengths[j]
			self.bitmasks[j+1:] = [k<<shift for k in self.bitmasks[j+1:]]
			self.shifts += [self.total_shift,]
			self.total_shift += shift
	def code_from_attributes(self, instance_number, *attributes):
		assert( isinstance(instance_number, int) )
		assert( instance_number>=0 )
		assert( len(attributes)==len(self.enumerations) )
		code = instance_number<<self.total_shift
		for j in range(len(self.enumerations)):
			if attributes[j] in self.enumerations[j]:
				code += attributes[j].value<<self.shifts[j]
			else:
				# pass value through enum to make sure it's valid
				code += self.enumerations[j](attributes[j]).value<<self.shifts[j]
		return code
	def attributes_from_code(self, code):
		attributes = (self.enumerations[j]((code&self.bitmasks[j])>>self.shifts[j]) for j in range(len(self.enumerations)))
		attributes = (code>>self.total_shift,) + tuple(attributes)
		return attributes
