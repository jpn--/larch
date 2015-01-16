.. currentmodule:: larch

=======================
Data Fundamentals
=======================

Larch requires data to be structured in one of two formats: the case-only (co)
format or the case-alternative (ca) format. This are commonly referred to as
IDCase (each record contains all the information for mode choice over
alternatives for a single trip) or IDCase-IDAlt (each record contains all the
information for a single alternative available to each decision maker so there is one
record for each alternative for each choice).

In the case-only format, each record provides all the relevant information
about an individual choice, including the variables related to the decision maker
or the choice itself, as well as alternative related variables for all available
alternatives and a variable indicating which alternative was chosen. In the
case-alternative format, each record can include information on the variables
related to the decision maker or the choice itself, the attributes of that
particular alternative, and a choice variable that indicates whether the
alternative was or was not chosen.

Unlike most other tools for discrete choice analysis, Larch does not demand you
employ one or the other of these data formats.  You can use either, or both
simultaneously.


