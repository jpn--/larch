.. currentmodule:: larch.model.latentclass

=======================
Latent Class Models
=======================

Larch is able to handle latent class models.  A latent class model can be thought of
as a two-stage model: first, to which class does an observation belong; and then, given
membership in that class, what are the choice probabilities.  Each class can have its
own distinct choice model, and parameters can be either independent or shared across
classes.

The class membership model is a choice model where "class membership" is the implied
choice, instead of the actual alternatives in the choice set.

Because the class membership model is not about the choices, it cannot use data about
the choices in the model definition -- the class membership model can be based only on
the `co` data.

A :class:`LatentClassModel` is the core object used to represent a latent class model.

.. autoclass:: LatentClassModel

