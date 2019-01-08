import os
import re
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, TreeBuilder, XMLParser
from contextlib import contextmanager
from .uid import uid as _uid
import base64
import cloudpickle, pickle
import pandas
from . import styles

# xml.etree.ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
# xml.etree.ElementTree.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# @import url(https://fonts.googleapis.com/css?family=Roboto+Mono:400,700,700italic,400italic,100,100italic);




# 	font-family: "Roboto Slab", Roboto, Helvetica, sans-serif;

_default_css = """

@import url(https://fonts.googleapis.com/css?family=Roboto:400,700,500italic,100italic|Roboto+Mono:300,400,700);

.error_report {color:red; font-family:monospace;}

body {""" + styles.body_font + """}

div.larch_title {
	font-family: "Book-Antiqua", "Palatino", serif;
	font-size:200%; 
	font-weight:900;
	font-style:normal; 
	color: #444444;
}

table {border-collapse:collapse;}

table, th, td {
	border: 1px solid #999999;
	font-family:"Roboto Mono", monospace;
	font-size:90%;
	font-weight:400;
	}

th, td { padding:2px; }

td.parameter_category {
	font-family:"Roboto", monospace;
	font-weight:500;
	background-color: #f4f4f4; 
	font-style: italic;
	}

th {
	font-family:"Roboto", monospace;
	font-weight:700;
	}

.larch_signature {""" + styles.signature_font + """}
.larch_name_signature {""" + styles.signature_name_font + """}

a.parameter_reference {font-style: italic; text-decoration: none}

.strut2 {min-width:1in}

.histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }

.dicta pre {
	margin:0;
	font-family:"Roboto Mono", monospace;
	font-weight:300;
	font-size:70%;
}

.raw_log pre {
	font-family:"Roboto Mono", monospace;
	font-weight:300;
	font-size:70%;
	}

caption {
    caption-side: bottom;
	text-align: left;
	font-family: Roboto;
	font-style: italic;
	font-weight: 100;
	font-size: 80%;
}

table.dictionary { border:0px hidden !important; border-collapse: collapse !important; }
div.blurb {
	margin-top: 15px;
	max-width: 6.5in;
}

div.note {
	font-size:90%;
	padding-left:1em;
	padding-right:1em;
	border: 1px solid #999999;
	border-radius: 4px;
}

p.admonition-title {
	font-weight: 700;
}

.tooltipped {
	position: relative;
	display: inline-block;
}

.tooltipped .tooltiptext {
	visibility: hidden;
	width: 180px;
	background-color: black;
	color: #fff;
	text-align: center;
	border-radius: 6px;
	padding: 5px 0;
	position: absolute;
	z-index: 1;
	top: -5px;
	left: 110%;
}

.tooltipped .tooltiptext::after {
	content: "";
	position: absolute;
	top: 50%;
	right: 100%;
	margin-top: -5px;
	border-width: 5px;
	border-style: solid;
	border-color: transparent black transparent transparent;
}
.tooltipped:hover .tooltiptext {
	visibility: visible;
}

"""

from xmle import Elem

def larch_style():
	return Elem(
		tag='style',
		text=_default_css,
	)




_tooltipped_style_css = """
		.tooltipped {
			position: relative;
			display: inline-block;
		}
		
		.tooltipped .tooltiptext {
			visibility: hidden;
			width: 180px;
			background-color: black;
			color: #fff;
			text-align: center;
			border-radius: 6px;
			padding: 5px 0;
			position: absolute;
			z-index: 1;
			top: -5px;
			left: 110%;
		}
		
		.tooltipped .tooltiptext::after {
			content: "";
			position: absolute;
			top: 50%;
			right: 100%;
			margin-top: -5px;
			border-width: 5px;
			border-style: solid;
			border-color: transparent black transparent transparent;
		}
		.tooltipped:hover .tooltiptext {
			visibility: visible;
		}
		


		
	"""


def tooltipped_style():
	return Elem('style', attrib={'type':"text/css"}, text=_tooltipped_style_css)



def floating_table_head():
	floatThead = Elem(tag="script", attrib={
		'src': "https://cdnjs.cloudflare.com/ajax/libs/floatthead/2.0.3/jquery.floatThead.min.js",
	})
	floatTheadA = Elem(tag="script", text = """
	$( document ).ready(function() {
		var $table = $('table.floatinghead');
		$table.floatThead({ position: 'absolute' });
		var $tabledf = $('table.dataframe');
		$tabledf.floatThead({ position: 'absolute' });
	});
	$(window).on("hashchange", function () {
		window.scrollTo(window.scrollX, window.scrollY - 50);
	});
	""")
	return floatThead, floatTheadA

