import ast
import operator as op

# supported operators
operator_strs = {ast.Add: ' + ', ast.Sub: ' - ', ast.Mult: ' * ',
             ast.Div: ' / ', ast.Pow: ' ** ', ast.BitXor: ' ^ ',
             ast.USub: ' -'}



def _repackage_comparison(a,op,b):
	"""
	Repackages certain operators for use in Alogit.
	"""
	if isinstance(op,ast.Eq):
		return "ifeq({},{})".format(repackage(a),repackage(b))
	elif isinstance(op,ast.Gt):
		return "ifgt({},{})".format(repackage(a),repackage(b))
	elif isinstance(op,ast.Lt):
		return "iflt({},{})".format(repackage(a),repackage(b))
	elif isinstance(op,ast.GtE):
		return "ifge({},{})".format(repackage(a),repackage(b))
	elif isinstance(op,ast.LtE):
		return "ifle({},{})".format(repackage(a),repackage(b))
	elif isinstance(op,ast.NotEq):
		return "ifne({},{})".format(repackage(a),repackage(b))
	elif isinstance(op,ast.In):
		rights = ",".join(repackage(j) for j in b.elts)
		return "if(ifeq({},{}))".format(repackage(a), rights)
	elif isinstance(op,ast.NotIn):
		rights = ",".join(repackage(j) for j in b.elts)
		return "not if(ifeq({},{}))".format(repackage(a), rights)
	else:
		global errop
		errop = op
		raise TypeError(op)



def repackage(expr):
	"""
	Repackages expressions for use in Alogit.
	"""
	if isinstance(expr, str):
		node = ast.parse(expr, mode='eval').body
	else:
		node = expr
	if isinstance(node, ast.Num): # <number>
		return "{!s}".format(node.n)
	elif isinstance(node, ast.BinOp): # <left> <operator> <right>
		return "({}{}{})".format(repackage(node.left), operator_strs[type(node.op)], repackage(node.right))
	elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
		return "({}{})".format(operator_strs[type(node.op)], repackage(node.operand))
	elif isinstance(node, ast.Name): # <operator> <operand> e.g., -1
		return "{!s}".format(node.id)
	elif isinstance(node, ast.Compare): # <operator> <operand> e.g., -1
		left = node.left
		right = node.comparators[0]
		op = node.ops[0]
		s = "("
		s += _repackage_comparison(left,op,right)
		for i in range(1,len(node.ops)):
			s += "*"
			left = right
			op = node.ops[i]
			right = node.comparators[i]
			s += _repackage_comparison(left,op,right)
		s += ")"
		return s
	else:
		global err
		err = node
		raise TypeError(node)
	
	
