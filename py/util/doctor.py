#### The doctor does not run a comprehensive check, there may be other issues too ####


import warnings

class LarchDoctorWarning(Warning):
	pass

def warn(msg):
	#warnings.warn(msg, LarchDoctorWarning, stacklevel=3)
	warnings.warn_explicit(msg, LarchDoctorWarning, __file__, 1)