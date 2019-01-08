import os
import math
import copy
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

class html_tables(object):

	def __init__(self, raw_html):
		self.url_soup = BeautifulSoup(raw_html, "lxml")

	def read(self):

		self.tables      = []
		self.tables_html = self.url_soup.find_all("table")

		# Parse each table
		for n in range(0, len(self.tables_html)):

			n_cols = 0
			n_rows = 0

			for row in self.tables_html[n].find_all("tr"):
				col_tags = row.find_all(["td", "th"])
				if len(col_tags) > 0:
					n_rows += 1
					if len(col_tags) > n_cols:
						n_cols = len(col_tags)

			# Create dataframe
			df = pd.DataFrame(index = range(0, n_rows), columns = range(0, n_cols))

			# Create list to store rowspan values
			skip_index = [0 for i in range(0, n_cols)]
			this_skip_index = copy.deepcopy(skip_index)

			# Start by iterating over each row in this table...
			row_counter = 0
			for row in self.tables_html[n].find_all("tr"):

				# Skip row if it's blank
				if len(row.find_all(["td", "th"])) == 0:
					pass

				else:

					# Get all cells containing data in this row
					columns = row.find_all(["td", "th"])
					col_dim = []
					row_dim = []
					col_dim_counter = -1
					row_dim_counter = -1
					col_counter = -1
					this_skip_index = copy.deepcopy(skip_index)

					for col in columns:

						# Determine cell dimensions
						colspan = col.get("colspan")
						if colspan is None:
							col_dim.append(1)
						else:
							col_dim.append(int(colspan))
						col_dim_counter += 1

						rowspan = col.get("rowspan")
						if rowspan is None:
							row_dim.append(1)
						else:
							row_dim.append(int(rowspan))
						row_dim_counter += 1

						# Adjust column counter
						if col_counter == -1:
							col_counter = 0
						else:
							col_counter = col_counter + col_dim[col_dim_counter - 1]

						try:
							while skip_index[col_counter] > 0:
								col_counter += 1
						except IndexError:
							from pprint import pprint
							print("~"*50)
							pprint(locals())
							print("~"*50)
							raise
							pass

						# Get cell contents
						cell_data = col.get_text()

						# Insert data into cell
						df.iat[row_counter, col_counter] = cell_data

						# Record column skipping index
						if row_dim[row_dim_counter] > 1:
							this_skip_index[col_counter] = row_dim[row_dim_counter]

				# Adjust row counter
				row_counter += 1

				# Adjust column skipping index
				skip_index = [i - 1 if i > 0 else i for i in this_skip_index]

			# Append dataframe to list of tables
			self.tables.append(df)

		return(self.tables)



def xml_table_to_txt(elem):
	first_table = html_tables(elem.tostring()).read()[0]
	return (first_table.to_string(
		buf=None, columns=None, col_space=None, header=True,
		index=False, na_rep='', formatters=None, float_format=None,
		sparsify=None, index_names=False, justify=None, line_width=None,
		max_rows=None, max_cols=None, show_dimensions=False).partition("\n")[2])
