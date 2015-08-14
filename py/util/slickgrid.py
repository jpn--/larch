from .temporaryfile import TemporaryFile
import os

_template = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
  <title>{title}</title>
  <link rel="stylesheet" href="{js_lib}/slick.grid.css" type="text/css"/>
  <link rel="stylesheet" href="{js_lib}/examples.css" type="text/css"/>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background-color: White;
      overflow: auto;
    }}

    body {{
      font: 11px Helvetica, Arial, sans-serif;
    }}

    #container {{
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
    }}

    #description {{
      position: fixed;
      top: 30px;
      right: 30px;
      width: 25em;
      background: beige;
      border: solid 1px gray;
      z-index: 1000;
    }}

    #description h2 {{
      padding-left: 0.5em;
    }}
  </style>
</head>
<body>
<div id="container"></div>

<script src="{js_lib}/lib/jquery-1.7.min.js"></script>
<script src="{js_lib}/lib/jquery.event.drag-2.2.js"></script>

<script src="{js_lib}/slick.core.js"></script>
<script src="{js_lib}/slick.grid.js"></script>
<script>
  var grid,
      data = [],
      columns = [
          {columns}
      ],
      options = {{
        enableCellNavigation: false,
        enableColumnReorder: false
      }};

  {data}

  grid = new Slick.Grid("#container", data, columns, options);
</script>
</body>
</html>
"""

_js_lib_default = os.path.join(os.path.dirname(__file__), 'slick_js')


def create_slickgrid_table(title="Untitled", js_lib=None, column_names=[], datarows=[]):
	if js_lib is None:
		js_lib = _js_lib_default
	each_column_def = """{{ id: "{}", name: "{}", field: "col{}", width: {} }}"""
	coldefs = []
	for c,col in enumerate(column_names):
		wid = 40
		if len(col)*7>wid: wid = len(col)*7
		coldefs.append(each_column_def.format(col.casefold().replace(' ','-'), col, c, wid))
	columns=",\n".join(coldefs)
	data = ""
	for n, datarow in enumerate(datarows):
		data_string = ",".join( '''col{0}: "{1!s}"'''.format(c, dataitem) for c, dataitem in enumerate(datarow))
		data += "data[{}]={{ {} }};\n".format(n, data_string)
	return _template.format(**locals())
	

def display_slickgrid(filename=None, **kwargs):
	t = TemporaryFile(suffix='.html')
	t.write(create_slickgrid_table(**kwargs))
	t.flush()
	t.view()
	
# print(sg.create_slickgrid_table('YOYOY', column_names=['A','B','C'], datarows=[(1,2,3),(4,5,6),(7,8,9)]))