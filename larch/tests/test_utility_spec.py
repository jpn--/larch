#

from larch import Model, P, X

def test_utility_function_output():
	k = Model()
	k.utility_ca = P.Aaa * X.Aaa + P.Bbb * X.Bbb + P.Ccc
	k.utility_co[1] = P.Dx1 + P.Dy1 * X.Yyy
	k.utility_co[2] = P.Dx2 + P.Dy2 * X.Yyy

	k.quantity_ca = P.Qaa * X.Aaa + P.Qbb * X.Bbb + P.Qcc

	k.set_values(Aaa=12, Bbb=20, Ccc=2, Dx1=0, Dy1=0.001, Dx2=0.33, Dy2=-0.002)

	u1 = k.utility_functions(resolve_parameters=False)

	assert u1.tostring() == '<div><table class="floatinghead" style="margin-top:1px;"><thead>' \
							'<tr><th>alt</th><th style="text-align:left;">formula</th></tr></thead>' \
							'<tbody><tr><td>1</td><td style="text-align:left;"><div></div> + ' \
							'<div class="tooltipped">P.Aaa<span class="tooltiptext">12</span></div> * ' \
							'<div class="tooltipped">X.Aaa<span class="tooltiptext">This is Data</span>' \
							'</div><br> + <div class="tooltipped">P.Bbb<span class="tooltiptext">20</span>' \
							'</div> * <div class="tooltipped">X.Bbb<span class="tooltiptext">This is Data' \
							'</span></div><br> + <div class="tooltipped">P.Ccc<span class="tooltiptext">2' \
							'</span></div><br> + <div class="tooltipped">P.Dx1<span class="tooltiptext">0' \
							'</span></div><br> + <div class="tooltipped">P.Dy1<span class="tooltiptext">0.001' \
							'</span></div> * <div class="tooltipped">X.Yyy<span class="tooltiptext">' \
							'This is Data</span></div><br> + log(<br>\xa0\xa0 + <span></span>exp(' \
							'<div class="tooltipped">P.Qaa<span class="tooltiptext">exp(0) = 0</span></div>) ' \
							'* <div class="tooltipped">X.Aaa<span class="tooltiptext">This is Data</span></div>' \
							'<br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">P.Qbb<span class="tooltiptext">' \
							'exp(0) = 0</span></div>) * <div class="tooltipped">X.Bbb<span class="tooltiptext">' \
							'This is Data</span></div><br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">' \
							'P.Qcc<span class="tooltiptext">exp(0) = 0</span></div>)<br>)</td></tr><tr><td>2</td>' \
							'<td style="text-align:left;"><div></div> + <div class="tooltipped">P.Aaa' \
							'<span class="tooltiptext">12</span></div> * <div class="tooltipped">X.Aaa' \
							'<span class="tooltiptext">This is Data</span></div><br> + <div class="tooltipped">' \
							'P.Bbb<span class="tooltiptext">20</span></div> * <div class="tooltipped">X.Bbb' \
							'<span class="tooltiptext">This is Data</span></div><br> + <div class="tooltipped">' \
							'P.Ccc<span class="tooltiptext">2</span></div><br> + <div class="tooltipped">P.Dx2' \
							'<span class="tooltiptext">0.33</span></div><br> + <div class="tooltipped">P.Dy2' \
							'<span class="tooltiptext">-0.002</span></div> * <div class="tooltipped">X.Yyy' \
							'<span class="tooltiptext">This is Data</span></div><br> + log(<br>\xa0\xa0 + ' \
							'<span></span>exp(<div class="tooltipped">P.Qaa<span class="tooltiptext">exp(0) = 0' \
							'</span></div>) * <div class="tooltipped">X.Aaa<span class="tooltiptext">' \
							'This is Data</span></div><br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">' \
							'P.Qbb<span class="tooltiptext">exp(0) = 0</span></div>) * <div class="tooltipped">' \
							'X.Bbb<span class="tooltiptext">This is Data</span></div><br>\xa0\xa0 + <span>' \
							'</span>exp(<div class="tooltipped">P.Qcc<span class="tooltiptext">exp(0) = 0</span>' \
							'</div>)<br>)</td></tr></tbody></table></div>'

	u2 = k.utility_functions(resolve_parameters=True)

	assert u2.tostring() == '<div><table class="floatinghead" style="margin-top:1px;"><thead><tr><th>alt</th>' \
							'<th style="text-align:left;">formula</th></tr></thead><tbody><tr><td>1</td>' \
							'<td style="text-align:left;"><div></div> + <div class="tooltipped">12' \
							'<span class="tooltiptext">P.Aaa</span></div> * <div class="tooltipped">' \
							'X.Aaa<span class="tooltiptext">This is Data</span></div><br> + <div class="tooltipped">' \
							'20<span class="tooltiptext">P.Bbb</span></div> * <div class="tooltipped">X.Bbb' \
							'<span class="tooltiptext">This is Data</span></div><br> + <div class="tooltipped">' \
							'2<span class="tooltiptext">P.Ccc</span></div><br> + <div class="tooltipped">0' \
							'<span class="tooltiptext">P.Dx1</span></div><br> + <div class="tooltipped">0.001' \
							'<span class="tooltiptext">P.Dy1</span></div> * <div class="tooltipped">X.Yyy' \
							'<span class="tooltiptext">This is Data</span></div><br> + log(<br>\xa0\xa0 + <span>' \
							'</span>exp(<div class="tooltipped">0<span class="tooltiptext">exp(P.Qaa)</span></div>) ' \
							'* <div class="tooltipped">X.Aaa<span class="tooltiptext">This is Data</span></div>' \
							'<br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">0<span class="tooltiptext">' \
							'exp(P.Qbb)</span></div>) * <div class="tooltipped">X.Bbb<span class="tooltiptext">' \
							'This is Data</span></div><br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">0' \
							'<span class="tooltiptext">exp(P.Qcc)</span></div>)<br>)</td></tr><tr><td>2</td>' \
							'<td style="text-align:left;"><div></div> + <div class="tooltipped">12' \
							'<span class="tooltiptext">P.Aaa</span></div> * <div class="tooltipped">X.Aaa' \
							'<span class="tooltiptext">This is Data</span></div><br> + <div class="tooltipped">' \
							'20<span class="tooltiptext">P.Bbb</span></div> * <div class="tooltipped">X.Bbb' \
							'<span class="tooltiptext">This is Data</span></div><br> + <div class="tooltipped">2' \
							'<span class="tooltiptext">P.Ccc</span></div><br> + <div class="tooltipped">0.33' \
							'<span class="tooltiptext">P.Dx2</span></div><br> + <div class="tooltipped">-0.002' \
							'<span class="tooltiptext">P.Dy2</span></div> * <div class="tooltipped">X.Yyy' \
							'<span class="tooltiptext">This is Data</span></div><br> + log(<br>\xa0\xa0 + <span>' \
							'</span>exp(<div class="tooltipped">0<span class="tooltiptext">exp(P.Qaa)</span></div>) ' \
							'* <div class="tooltipped">X.Aaa<span class="tooltiptext">This is Data</span></div>' \
							'<br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">0<span class="tooltiptext">' \
							'exp(P.Qbb)</span></div>) * <div class="tooltipped">X.Bbb<span class="tooltiptext">' \
							'This is Data</span></div><br>\xa0\xa0 + <span></span>exp(<div class="tooltipped">' \
							'0<span class="tooltiptext">exp(P.Qcc)</span></div>)<br>)</td></tr></tbody></table></div>'

