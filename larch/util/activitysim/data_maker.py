import os
import subprocess


def make_estimation_data(directory="test"):
	if os.path.exists(directory):
		check_file = os.path.join(directory, ".gitignore")
		if not os.path.exists(check_file):
			import importlib
			asim_create = importlib.import_module("activitysim.cli.create")
			asim_create.get_example("example_estimation_sf", directory)
			os.chdir(directory)
			cp = subprocess.run(
				[
					"activitysim",
					"run",
					"-c",
					"configs_estimation/configs",
					"-c",
					"configs",
					"-o",
					"output",
					"-d",
					"data_sf",
				],
				capture_output=True,
			)
			with open(check_file, 'rt') as f:
				f.write("**/*.csv\n")
				f.write("**/*.txt\n")
				f.write("**/*.yaml\n")
				f.write("**/.gitignore\n")
		else:
			print(f"using existing directory `{directory}`")
			os.chdir(directory)
