
body_font = 'font-family: "Book-Antiqua", "Palatino", serif;'

signature_font = 'font-size:70%; font-weight:100; font-style:italic; font-family: Roboto, Helvetica, sans-serif;'

signature_name_font = 'font-weight:400; font-style:normal; font-family: "Roboto Slab", Roboto, Helvetica, sans-serif;'




def load_css(filename):
	import os
	css = None
	if filename is None or not isinstance(filename, str):
		return None
	if os.path.exists(filename):
		with open(filename, 'r') as f:
			css = f.read()
		return css
	f0 = "{}.css".format(filename)
	if os.path.exists(f0):
		with open(f0, 'r') as f:
			css = f.read()
		return css
	try:
		import appdirs
	except ImportError:
		pass
	else:
		f1 = os.path.join(appdirs.user_config_dir('Larch'), filename)
		if os.path.exists(f1):
			with open(f1, 'r') as f:
				css = f.read()
			return css
		f2 = "{}.css".format(f1)
		if os.path.exists(f2):
			with open(f2, 'r') as f:
				css = f.read()
			return css
	if '{' in filename and '}' in filename:
		return filename
	return css


_default_css_jupyter = """

@import url('https://fonts.googleapis.com/css?family=Roboto:400,700,500italic,100italic|Roboto+Mono:300,400,700|EB+Garamond:400,400i');
@import url('https://fonts.googleapis.com/css?family=Roboto+Slab:100,400,700&display=swap');

.error_report {
	color:red; font-family:monospace;
}

div.output_wrapper {""" + body_font + """}

div.output_wrapper table,
div.jp-OutputArea-output table
{	
	border-collapse:collapse;
}

div.output_wrapper table, div.output_wrapper th, div.output_wrapper td,
div.jp-OutputArea-output table, div.jp-OutputArea-output th, div.jp-OutputArea-output td 
{
	border: 1px solid #999999;
	font-family:"Roboto Mono", monospace;
	font-size:9pt;
	font-weight:400;
}

div.output_wrapper th, div.output_wrapper td,
div.jp-OutputArea-output th, div.jp-OutputArea-output td
{ 
	padding:2px; text-align:left; 
}

div.output_wrapper td.parameter_category,
div.jp-OutputArea-output td.parameter_category
{
	font-family:"Roboto", monospace;
	font-weight:500;
	background-color: #f4f4f4; 
	font-style: italic;
}

div.output_wrapper th,
div.jp-OutputArea-output th
{
	font-family:"Roboto", monospace;
	font-weight:700;
}

div.output_wrapper table.dicta,
div.jp-OutputArea-output table.dicta
{ 
	border-left: 2px solid black; margin-bottom:2px; border-top:0; border-right:0
}

div.output_wrapper th.dicta, div.output_wrapper td.dicta,
div.jp-OutputArea-output th.dicta, div.jp-OutputArea-output td.dicta 
{ 
	padding-top:0px; text-align:left; border:0; 
}

div.output_wrapper div.LinearFunc,
div.jp-OutputArea-output div.LinearFunc
{
	font-family:"Roboto Mono", monospace;
	font-size:100%;
	font-weight:400;
}



.larch_signature {""" + signature_font + """ }
.larch_name_signature {""" + signature_name_font + """}

.larch_head_tag {font-size:150%; font-weight:900; font-family:"Roboto Slab", "Roboto", Verdana;}
.larch_head_tag_ver {font-size:80%; font-weight:200; font-family:"Roboto Slab", "Roboto", Verdana;}
.larch_head_tag_pth {font-size:40%; font-weight:200; font-family:"Roboto Slab", "Roboto", Verdana; padding-left:5px;}
.larch_head_tag_more {font-size:50%; font-weight:300; font-family:"Roboto Mono", monospace; line-height:130%;}

div.output_wrapper a.parameter_reference,
div.jp-OutputArea-output a.parameter_reference
{
	font-style: italic; text-decoration: none
}

div.output_wrapper .strut2, 
div.jp-OutputArea-output .strut2 
{
	min-width:1in
}

div.output_wrapper .histogram_cell,
div.jp-OutputArea-output .histogram_cell 
{ 
	padding-top:1; padding-bottom:1; vertical-align:center; 
}

div.output_wrapper .raw_log pre,
div.jp-OutputArea-output .raw_log pre
{
	font-family:"Roboto Mono", monospace;
	font-weight:300;
	font-size:70%;
}

.dicta pre 
{
	margin:0;
}

div.output_wrapper caption,
div.jp-OutputArea-output caption,
{
    caption-side: bottom;
	text-align: left;
	font-family: Roboto;
	font-style: italic;
	font-weight: 100;
	font-size: 80%;
}

table.running_parameter_update caption 
{
    caption-side: top;
	text-align: left;
	font-family: Roboto;
	font-style: italic;
	font-weight: 500;
	font-size: 100%;
}

table.dictionary 
{ 
	border:0px hidden !important; border-collapse: collapse !important; 
}

div.blurb {
	margin-top: 15px;
	max-width: 6.5in;
}

h2.figure_head {
	padding-left: .25in;
}

h3.figure_head {
	padding-left: .5in;
}

div.jp-RenderedMarkdown h1 {font-weight: 900; border-bottom:2px black solid; padding-bottom:4px}
div.jp-RenderedMarkdown h2 {font-weight: 850; border-bottom:0.5px black solid; padding-bottom:4px}
div.jp-RenderedMarkdown h3 {font-weight: 800; font-style:italic}

div.jp-RenderedMarkdown p {
    max-width:600px;
    font-size:150%;
    font-family: EB Garamond;
}

div.jp-RenderedMarkdown p code {
    font-size: 75%;
}


"""

from .display import display_html, HTML
from .xhtml import tooltipped_style, floating_table_head, _tooltipped_style_css

css = HTML("<style>{}\n\n{}</style>".format(_default_css_jupyter,tooltipped_style().tostring()))

def stylesheet():
	display_html(css)
	# floatThead, floatTheadA = floating_table_head()
	# display_html(HTML(floatThead.tostring()))
	# display_html(HTML(floatTheadA.tostring()))




