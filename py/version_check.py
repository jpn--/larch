


def outdated():
	from pip.commands import list as list_
	list_command = list_.ListCommand()
	options, args = list_command.parse_args(['--outdated'])
	packages = list_command.find_packages_latests_versions(options)
	for dist, remote_version in packages:
		if remote_version > dist.parsed_version:
			if dist.project_name == "larch":
				print("Version {} of Larch is now available (you have {})".format(remote_version,dist.parsed_version))

