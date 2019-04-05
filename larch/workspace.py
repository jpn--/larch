
from xmle import Reporter, NumberedCaption

def make_reporter(title, notes=True, viz=True):
	"""
	Make a standard reporter.
	"""

	REPORT = Reporter(title)
	REPORT.MAIN = REPORT.section("Estimated Model", "Estimated Model")
	if notes is True:
		REPORT.NOTES = REPORT.section("Model Notes", "Notes")
	elif notes:
		REPORT.NOTES = REPORT.section(notes, notes)
	if viz is True:
		REPORT.VIZ = REPORT.section("Model Visualization", "Visualization")
	elif viz:
		REPORT.VIZ = REPORT.section(viz, viz)

	REPORT.FIG = NumberedCaption('Figure')
	REPORT.TAB = NumberedCaption('Table')

	return REPORT

