


import warnings

class LarchDoctorWarning(Warning):
	pass

def warn(msg):
	warnings.warn(msg, LarchDoctorWarning, stacklevel=3)