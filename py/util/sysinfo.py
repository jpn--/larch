import platform, subprocess, re, resource

def get_processor_name():
	"""Get a descriptive name of the CPU on this computer"""
	if platform.system() == "Windows":
		return platform.processor()
	elif platform.system() == "Darwin":
		command =("sysctl", "-n", "machdep.cpu.brand_string")
		return subprocess.check_output(command).strip()
	elif platform.system() == "Linux":
		command = ("cat", "/proc/cpuinfo")
		all_info = subprocess.check_output(command, shell=True).strip()
		for line in all_info.split("\n"):
			if "model name" in line:
				return re.sub( ".*model name.*:", "", line,1)
	return ""

def get_peak_memory_usage():
	mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	if mem > 2.0*2**30:
		return str(mem/2**30) + " GiB"
	return str(mem/2**20) + " MiB"
