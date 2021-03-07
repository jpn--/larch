from .model import NumbaModel as Model
top = __import__(__name__.split('.')[0])
top.Model = Model
