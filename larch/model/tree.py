import heapq
from collections import OrderedDict

import networkx as nx
import numpy as np

from ..util.lazy import lazy


class NestingTree(nx.DiGraph):

    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict

    def __get__(self, instance, owner):
        # self : SubkeyStore
        # instance : instance of parent class that has `self` as a member, or None
        # owner : class of `instance`
        if instance is None:
            pass  # print("GRR: no instance")
            return self
        newself = getattr(instance, self.private_name, None)
        if newself is None:
            pass  # print(f"GRR No Current: {instance=} {owner=}")
            try:
                instance.initialize_graph()
            except ValueError:
                pass
            newself = getattr(instance, self.private_name, None)
        if newself is not None:
            newself._instance = instance
        pass  # print(f"GRR: get {instance=} {newself=}")
        return newself

    def __set__(self, instance, value):
        # self : NestingTree object
        # instance : instance of parent class that has `self` as a member
        # value : the new value that is trying to be assigned
        assert isinstance(value, NestingTree)
        t = value.copy()
        t._instance = instance
        setattr(instance, self.private_name, t)
        try:
            t._instance.mangle()
        except AttributeError as err:
            pass  # print(f"GRR: {err}")
        else:
            pass  # print(f"GRR Mangle: {instance}")

    def __delete__(self, instance):
        setattr(instance, self.private_name, None)
        try:
            instance.mangle()
        except AttributeError as err:
            pass  # print(f"GRR: {err}")
        else:
            pass  # print(f"GRR Mangle: {instance}")

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_private_" + name

    def touch(self):
        try:
            self._instance.mangle()
        except AttributeError:
            pass  # print("GRR: mangle failure")
        else:
            pass  # print("GRR: mangle ok")

    def __init__(self, *arg, root_id=0, suggested_elemental_order=(), **kwarg):
        if len(arg) and isinstance(arg[0], NestingTree):
            super().__init__(*arg, **kwarg)
            self._root_id = arg[0]._root_id
            if suggested_elemental_order != ():
                self._suggested_elemental_order = suggested_elemental_order
            else:
                self._suggested_elemental_order = arg[0]._suggested_elemental_order
        else:
            super().__init__(*arg, **kwarg)
            self._root_id = root_id
            self._suggested_elemental_order = suggested_elemental_order
        if self._root_id not in self.nodes:
            self.add_node(root_id, name="_root_", root=True)
        self._clear_caches()

    def __eq__(self, other):
        return (
            self._adj == other._adj
            and self._node == other._node
            and self._root_id == other._root_id
            and self._suggested_elemental_order == other._suggested_elemental_order
        )

    def suggest_elemental_order(self, order):
        self._suggested_elemental_order = tuple(j for j in order if j in self.nodes)

    @property
    def root_id(self):
        """int : The code for the root node."""
        return self._root_id

    @root_id.setter
    def root_id(self, x):
        top_nests = list(self.successors(self._root_id))
        top_attrs = [self.edges[self._root_id, t] for t in top_nests]
        if self._root_id in self.nodes:
            self.remove_node(self._root_id)
        self._root_id = x
        if self._root_id not in self.nodes:
            self.add_node(self._root_id, name="_root_", root=True)
        for t, a in zip(top_nests, top_attrs):
            self.add_edge(self._root_id, t, **a, _clear_caches=False)
        self._clear_caches()

    def _clear_caches(self):
        NestingTree.topological_sorted.invalidate(self, "topological_sorted")
        NestingTree.topological_sorted_no_elementals.invalidate(
            self, "topological_sorted_no_elementals"
        )
        NestingTree.standard_sort.invalidate(self, "standard_sort")
        NestingTree.standard_sort.invalidate(self, "standard_slot_map")
        NestingTree.elementals.invalidate(self, "elementals")
        NestingTree.standard_competitive_edge_list.invalidate(
            self, "standard_competitive_edge_list"
        )
        NestingTree.standard_competitive_edge_list_2.invalidate(
            self, "standard_competitive_edge_list_2"
        )
        self._predecessor_slots = {}
        self._successor_slots = {}
        self.touch()

    def add_edge(self, u, v, implied=False, _clear_caches=True, **kwarg):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords.

        Parameters
        ----------
        u, v : int
                Nodes should be integer codes. The upstream node `u` is
                a nest or the root node.  Downsteam node `v` can be
                a nest or elemental alternative.
        implied : bool, default False
                Implied edges are for connection of otherwise unconnected
                nests to the root node.
        _clear_caches : bool, default True
        kwarg : keyword arguments, optional
                Edge data (or labels or objects) can be assigned using
                keyword arguments.
        """
        if not implied:
            drops = []
            for u_, v_, imp_ in self.in_edges(nbunch=[v], data="implied"):
                if imp_:
                    drops.append([u_, v_])
            for d in drops:
                super().remove_edge(*d)
        if _clear_caches:
            self._clear_caches()
        return super().add_edge(int(u), int(v), implied=implied, **kwarg)

    def _remove_edge_no_implied(self, u, v, *arg, **kwarg):
        result = super().remove_edge(u, v)
        self._clear_caches()
        return result

    def remove_edge(self, u, v, *arg, **kwarg):
        """
        Remove the edge between u and v.

        Parameters
        ----------
        u, v : int
                Remove the edge between nodes u and v.

        Raises
        ------
        NetworkXError
                If there is not an edge between u and v.
        """
        result = super().remove_edge(u, v)
        if self.in_degree(v) == 0 and v != self._root_id:
            self.add_edge(self._root_id, v, implied=True, _clear_caches=False)
        self._clear_caches()
        return result

    def add_node(
        self,
        code,
        *,
        children=(),
        parent=None,
        parents=None,
        phi_parameters=None,
        **kwarg,
    ):
        """
        Add a single node `code` and update node attributes.

        Parameters
        ----------
        code : int
                Although the generic networkx.DiGraph allows a node
                to be any hashable Python object except None, Larch
                assumes that node codes are integers.
        children : Collection
                A collection of other node codes that are the children
                of this new node.  Links will be created from this node
                to each child.
        parent : int, optional
                The parent of this new node. If not given, the root
                node is assumed to be the parent of this node, and an
                implied link is created.  This implied link is removed
                if the node is later made the child of some other node.
                If the parent is set explicitly, the link is *not*
                removed later.
        parents : Collection, optional
                Set multiple parent up-stream nodes.
        phi_parameters : Mapping
                Set phi parameters on graph links connecting to this
                node, used in network GEV models. The keys of this mapping
                indicate the node at the other end of the link, and the
                values are parameter names.
        kwarg : other keyword arguments, optional
                Set or change node attributes using key=value.
        """
        if parents is not None and parent is not None:
            raise TypeError("cannot give both parent and parents arguments")
        super().add_node(code, **kwarg)
        for child in children:
            self.add_edge(code, child, _clear_caches=False)
        if parent is not None:
            self.add_edge(parent, code, _clear_caches=False)
        elif parents is not None:
            for p in parents:
                self.add_edge(p, code, _clear_caches=False)
        else:
            if self.in_degree(code) == 0 and code != self._root_id:
                self.add_edge(self._root_id, code, implied=True, _clear_caches=False)
        if phi_parameters is not None:
            for k, parametername in phi_parameters.items():
                if (code, k) in self.edges:
                    self.edges[code, k]["parameter"] = str(parametername)
                elif (k, code) in self.edges:
                    self.edges[k, code]["parameter"] = str(parametername)
                else:
                    raise ValueError(
                        f"connected node {k} from phi_parameters not found"
                    )
        self._clear_caches()

    def new_node(self, *, code=None, **kwarg):
        """
        Add a new nesting node to this NestingTree.

        A new unique code is automatically created and returned by
        this method for creating new nests.

        All arguments must be given as keyword parameters.

        Parameters
        ----------
        parameter : str
                The name of the parameter to associate with this nest.
        children : Collection[int], optional
                The code numbers for the children of this nest.  These can be
                elemental alternatives or other nests.  If not given, no children
                will be defined initially, but they can be added later.
        parent : int, optional
                The code number for the parent of this nest.  If not given,
                the parent is implied as the root node, unless and until set
                to some other node.
        name : str, optional
                A human-readable name to associate with this nest.
        code : int, optional
                Use this code for the new nest. If this code already exists,
                a ValueError is raised.

        Returns
        -------
        int
                The new code for this nest.

        Raises
        ------
        ValueError
                If a new code is given but it already exists in this tree.
        """
        if code is None:
            proposed_code = len(self)
            while proposed_code in self:
                proposed_code += 1
        else:
            if code in self:
                raise ValueError(f"code {code} already exists in this tree")
            proposed_code = code
        self.add_node(proposed_code, **kwarg)
        return proposed_code

    def add_nodes(self, codes, *arg, parent=None, **kwarg):
        for code in codes:
            self.add_node(code, *arg, parent=parent, **kwarg)

    def remove_node(self, n):
        """
        Remove node n.

        Removes the node n, reconnecting all outedges to the head node
        of all inedges. Attempting to remove a non-existent node will
        raise an exception.

        Parameters
        ----------
        n : int
                A node in the graph
        """
        replace_edges = {k: self.edges[k].copy() for k in self.edges(n)}
        replace_heads = [k for k, _ in self.in_edges(n)]
        super(NestingTree, self).remove_node(n)
        for k, attrs in replace_edges.items():
            for h in replace_heads:
                super().add_edge(h, k[1], **attrs)
        self._clear_caches()

    @lazy
    def topological_sorted(self):
        return list(reverse_lexicographical_topological_sort(self))

    @lazy
    def topological_sorted_no_elementals(self):
        try:
            result = self.topological_sorted.copy()
        except nx.NetworkXUnfeasible:
            from networkx.algorithms.cycles import find_cycle

            try:
                cycle = find_cycle(self)
            except:
                pass
            else:
                print("Found graph cycle:")
                print(list(cycle))
            raise

        # collect zero out-degree (elemental) codes in a set
        # to remove them in a batch, which is much faster than
        # removing them from the list one at a time.
        to_remove = set()
        for code, out_degree in self.out_degree:
            if not out_degree:
                to_remove.add(code)
        return [i for i in result if i not in to_remove]

    @lazy
    def standard_sort(self):
        return self.elementals + tuple(self.topological_sorted_no_elementals)

    def node_name(self, code):
        return self.nodes[code].get("name", str(code))

    @property
    def standard_sort_names(self):
        return [self.node_name(s) for s in self.standard_sort]

    def node_names(self):
        return {s: (self.node_name(s) or s) for s in self.standard_sort}

    def elemental_names(self):
        return {s: (self.node_name(s) or s) for s in self.elementals}

    @lazy
    def standard_slot_map(self):
        return {i: n for n, i in enumerate(self.standard_sort)}

    def predecessor_slots(self, code):
        if code in self._predecessor_slots:
            return self._predecessor_slots[code]
        s = np.empty(self.in_degree(code), dtype=np.int32)
        for n, i in enumerate(self.predecessors(code)):
            s[n] = self.standard_slot_map[i]
        self._predecessor_slots[code] = s
        return s

    def successor_slots(self, code):
        if code in self._successor_slots:
            return self._successor_slots[code]
        s = np.empty(self.out_degree(code), dtype=np.int32)
        for n, i in enumerate(self.successors(code)):
            s[n] = self.standard_slot_map[i]
        self._successor_slots[code] = s
        return s

    def __elementals_iter(self):
        for code, out_degree in self.out_degree:
            if not out_degree:
                yield code

    @lazy
    def elementals(self):
        result = []
        found = set()
        for e in self._suggested_elemental_order:
            if self.out_degree(e) == 0:
                result.append(e)
                found.add(e)
        for e in sorted(self.__elementals_iter()):
            if e not in found:
                result.append(e)
                found.add(e)
        return tuple(result)

    def n_elementals(self):
        return len(self.elementals)

    def n_intermediate_nests(self):
        return len(self.nodes) - self.n_elementals() - 1

    def elemental_descendants_iter(self, code):
        if not self.out_degree(code):
            yield code
            return
        all_d = nx.descendants(self, code)
        for dcode, dout_degree in self.out_degree(all_d):
            if not dout_degree:
                yield dcode

    def elemental_descendants(self, code):
        return [i for i in self.elemental_descendants_iter(code)]

    @property
    def n_edges(self):
        return self.number_of_edges()

    def edge_slot_arrays(self, alpha_locator=None):
        s = self.n_edges
        up = np.zeros(s, dtype=np.int32)
        dn = np.zeros(s, dtype=np.int32)
        first_visit = np.zeros(s, dtype=np.int32)
        alloc_slot = np.full_like(first_visit, -1)
        n = s
        first_visit_found = set()
        for upcode in reversed(self.standard_sort):
            upslot = self.standard_slot_map[upcode]
            for dnslot in reversed(self.successor_slots(upcode)):
                n -= 1
                up[n] = upslot
                dn[n] = dnslot
        for n in range(s):
            if dn[n] not in first_visit_found:
                first_visit[n] = 1
                first_visit_found.add(dn[n])
        if alpha_locator is not None:
            for n in range(s):
                alloc_slot[n] = alpha_locator.get(
                    (self.standard_sort[up[n]], self.standard_sort[dn[n]]), -1
                )
        return up, dn, first_visit, alloc_slot

    def nodes_with_successors_iter(self):
        for code, out_degree in self.out_degree:
            if out_degree:
                yield code

    def nodes_with_multiple_predecessors_iter(self):
        for code, in_degree in self.in_degree:
            if in_degree > 1:
                yield code

    @lazy
    def standard_competitive_edge_list(self):
        alphas = []
        for n in self.nodes_with_multiple_predecessors_iter():
            predecessors = sorted(self.predecessors(n))
            for k in predecessors:
                alphas.append((k, n))
        return alphas

    @lazy
    def standard_competitive_edge_list_2(self):
        alphas = []
        for n in self.nodes_with_multiple_predecessors_iter():
            predecessors = sorted(self.predecessors(n))
            alphas.append((predecessors, n))
        return alphas

    def __getstate__(self):
        attr = {}  # self.__dict__.copy()
        no_pickle = (
            "topological_sorted",
            "topological_sorted_no_elementals",
            "standard_sort",
            "standard_slot_map",
            #'_standard_elemental_sort',
            "elementals",
            "_predecessor_slots",
            "_successor_slots",
            "_touch",
            "node_dict_factory",
            "_instance",
        )
        for k, v in self.__dict__.items():
            if k not in no_pickle:
                attr[k] = v
        return attr

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self._predecessor_slots = {}
        self._successor_slots = {}

    def __xml__(self, use_viz=True, use_dot=True, output="svg", figsize=None, **format):
        viz = None
        dot = None
        if use_viz:
            try:
                import pygraphviz as viz
            except ImportError:
                if use_dot:
                    try:
                        import pydot as dot
                    except ImportError:
                        pass
        elif use_dot:
            try:
                import pydot as dot
            except ImportError:
                pass

        if viz is None and dot is None:
            import warnings

            if use_viz and use_dot:
                msg = "neither pydot nor pygraphviz modules are installed, unable to draw nesting tree"
            elif use_viz:
                msg = "pygraphviz module not installed, unable to draw nesting tree"
            elif use_dot:
                msg = "pydot module not installed, unable to draw nesting tree"
            else:
                msg = "no drawing module used, unable to draw nesting tree"
            warnings.warn(msg)
            raise NotImplementedError(msg)

        if viz is not None:
            existing_format_keys = list(format.keys())
            for key in existing_format_keys:
                if key.upper() != key:
                    format[key.upper()] = format[key]
            if "SUPPRESSGRAPHSIZE" not in format:
                if "GRAPHWIDTH" not in format:
                    format["GRAPHWIDTH"] = 6.5
                if "GRAPHHEIGHT" not in format:
                    format["GRAPHHEIGHT"] = 4
            if "UNAVAILABLE" not in format:
                format["UNAVAILABLE"] = True
            # x = XML_Builder("div", {'class':"nesting_graph larch_art"})
            # x.h2("Nesting Structure", anchor=1, attrib={'class':'larch_art_xhtml'})
            from io import BytesIO

            if "SUPPRESSGRAPHSIZE" not in format:
                G = viz.AGraph(
                    name="Tree",
                    directed=True,
                    size="{GRAPHWIDTH},{GRAPHHEIGHT}".format(**format),
                )
            else:
                G = viz.AGraph(name="Tree", directed=True)
            for n in self.nodes:
                nname = self.nodes[n].get("name", n)
                if nname == n:
                    G.add_node(
                        n,
                        label="<{1}>".format(n, nname),
                        style="rounded,solid",
                        shape="box",
                    )
                else:
                    G.add_node(
                        n,
                        label='<{1} <FONT COLOR="#999999">({0})</FONT>>'.format(
                            n, nname
                        ),
                        style="rounded,solid",
                        shape="box",
                    )
            eG = G.add_subgraph(
                name="cluster_elemental",
                nbunch=self.elementals,
                color="#cccccc",
                bgcolor="#eeeeee",
                label="Elemental Alternatives",
                labelloc="b",
                style="rounded,solid",
            )
            unavailable_nodes = set()
            # if format['UNAVAILABLE']:
            # 	if self.is_provisioned():
            # 		try:
            # 			for n, ncode in enumerate(self.alternative_codes()):
            # 				if np.sum(self.Data('Avail'),axis=0)[n,0]==0: unavailable_nodes.add(ncode)
            # 		except: raise
            # 	try:
            # 		legible_avail = not isinstance(self.df.queries.avail, str)
            # 	except:
            # 		legible_avail = False
            # 	if legible_avail:
            # 		for ncode,navail in self.df.queries.avail.items():
            # 			try:
            # 				if navail=='0': unavailable_nodes.add(ncode)
            # 			except: raise
            # 	eG.add_subgraph(name='cluster_elemental_unavailable', nbunch=unavailable_nodes, color='#bbbbbb', bgcolor='#dddddd',
            # 				   label='Unavailable Alternatives', labelloc='b', style='rounded,solid')
            G.add_node(self.root_id, label="Root")
            up_nodes = set()
            down_nodes = set()
            for i, j in self.edges:
                G.add_edge(i, j)
                down_nodes.add(j)
                up_nodes.add(i)
            pyg_imgdata = BytesIO()
            try:
                G.draw(
                    pyg_imgdata, format=output, prog="dot"
                )  # write postscript in k5.ps with neato layout
            except ValueError as err:
                if "in path" in str(err):
                    import warnings

                    warnings.warn(str(err) + "; unable to draw nesting tree in report")
                    raise NotImplementedError()
            if output == "svg":
                import xml.etree.ElementTree as ET

                ET.register_namespace("", "http://www.w3.org/2000/svg")
                ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
                return ET.fromstring(pyg_imgdata.getvalue().decode())
            else:
                raise NotImplementedError(f"output {output} with use_viz")
        else:

            pydot = dot

            # set Graphviz graph type
            if self.is_directed():
                graph_type = "digraph"
            else:
                graph_type = "graph"
            strict = nx.number_of_selfloops(self) == 0 and not self.is_multigraph()

            name = self.name
            graph_defaults = self.graph.get("graph", {})
            if name == "":
                P = pydot.Dot(
                    "", graph_type=graph_type, strict=strict, **graph_defaults
                )
            else:
                P = pydot.Dot(
                    '"%s"' % name,
                    graph_type=graph_type,
                    strict=strict,
                    **graph_defaults,
                )
            try:
                P.set_node_defaults(**self.graph["node"])
            except KeyError:
                pass
            try:
                P.set_edge_defaults(**self.graph["edge"])
            except KeyError:
                pass

            cluster_elemental = pydot.Cluster(
                "elemental",
                style="rounded",
                bgcolor="lightgrey",
                color="white",
                rank="same",
                rankdir="LR",
            )

            for n, nodedata in self.nodes(data=True):
                str_nodedata = dict(
                    (k if k != "name" else "name_", '"' + str(v) + '"')
                    for k, v in nodedata.items()
                )

                if "parameter" in nodedata:
                    param_label = '<BR ALIGN="CENTER" /><FONT COLOR="#999999" POINT-SIZE="9"><I>{0}</I></FONT>'.format(
                        nodedata["parameter"]
                    )
                else:
                    param_label = ""

                if "name" in nodedata and n != self.root_id:
                    name = nodedata["name"]
                    str_nodedata["label"] = (
                        "<"
                        '<FONT COLOR="#999999" POINT-SIZE="9">({1}) </FONT>'
                        "{0}"
                        "{2}>".format(name, n, param_label)
                    )

                # Default styling for nodes can have been overridden
                if n in self.elementals:
                    str_nodedata["style"] = str_nodedata.get("style", "filled")
                    str_nodedata["fillcolor"] = str_nodedata.get("fillcolor", "white")
                elif n == self.root_id:
                    str_nodedata["shape"] = str_nodedata.get("shape", "invhouse")
                else:
                    str_nodedata["style"] = str_nodedata.get("style", "rounded")
                    str_nodedata["shape"] = str_nodedata.get("shape", "rectangle")

                p = pydot.Node(str(n), **str_nodedata)
                P.add_node(p)
                if n in self.elementals:
                    cluster_elemental.add_node(p)

            P.add_subgraph(cluster_elemental)

            if self.is_multigraph():
                for u, v, key, edgedata in self.edges(data=True, keys=True):
                    str_edgedata = dict(
                        (k, str(v_)) for k, v_ in edgedata.items() if k != "key"
                    )
                    if v in self.elementals:
                        str_edgedata["constraint"] = "false"
                    edge = pydot.Edge(str(u), str(v), key=str(key), **str_edgedata)
                    P.add_edge(edge)

            else:
                for u, v, edgedata in self.edges(data=True):
                    str_edgedata = dict(
                        (k, '"' + str(v) + '"') for k, v in edgedata.items()
                    )
                    edge = pydot.Edge(str(u), str(v), **str_edgedata)
                    P.add_edge(edge)

            ###
            from xmle import Elem

            prog = None
            if output == "svg":
                import xml.etree.ElementTree as ET

                ET.register_namespace("", "http://www.w3.org/2000/svg")
                ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
            elif output == "png":
                prog = [P.prog, "-Gdpi=300"]
                if figsize is not None:
                    prog.append(f"-Gsize={figsize[0]},{figsize[1]}\!")
                e = Elem.from_any(P.create(prog=prog, format=output, **format))
                e.attrib["dpi"] = (300, 300)
                return e
            return Elem.from_any(P.create(prog=prog, format=output, **format))

    def _repr_html_(self):
        from xmle import Elem

        x = Elem("div") << (self.__xml__())
        return x.tostring()

    def to_png(self, figsize=None, filename=None):
        """
        Output the graph visualization as a png.

        Parameters
        ----------
        figsize : 2-tuple, optional
                The (width, height) in inches.

        Returns
        -------
        xmle.Elem
        """
        result = self.__xml__(output="png", use_viz=False, figsize=figsize)
        if filename is not None:
            import base64

            if result.attrib["src"][:22] != "data:image/png;base64,":
                raise ValueError(
                    "problem decoding png:{}".format(result.attrib["src"][:22])
                )
            with open(filename, "wb") as fh:
                fh.write(base64.decodebytes(result.attrib["src"][22:].encode()))
        return result

    def partial_figure(
        self, including_nodes=None, source=None, *, n=None, n_at_level=3, n_expand=1
    ):
        """
        Generate a partial figure of the graph.

        Parameters
        ----------
        including_nodes : iterable or None
                An iterable containing node codes or names (or a mix).
        source : nodecode, optional
                All paths from this node to everything in `including_nodes` will be represented.
                Defaults to `root_id`.
        n : int
                If including_nodes is None, select this number of nodes randomly

        Returns
        -------
        Elem
        """
        if source is None:
            source = self.root_id
        from networkx.algorithms.simple_paths import all_simple_paths

        shows = set()

        if including_nodes is None and n is not None:
            including_nodes = sorted(np.random.choice(self.nodes, n, replace=False))

        if including_nodes is None and n_at_level is not None:
            import itertools
            from collections import deque

            q = deque([self.root_id])
            including_nodes = []
            while q:
                i = q.popleft()
                take = tuple(itertools.islice(self.successors(i), n_at_level))
                q.extend(take[:n_expand])
                including_nodes.extend(take)

        # Add every node in every path from the root to each `including_nodes`
        for each_node in including_nodes:
            if each_node in self.nodes:
                for i in all_simple_paths(self, source, each_node):
                    for j in i:
                        shows.add(j)
            else:
                for each_node_ in self.get_nodes_by_name(each_node):
                    for i in all_simple_paths(self, source, each_node_):
                        for j in i:
                            shows.add(j)
        s = self.subgraph(shows)
        return graph_to_figure(s)

    def get_nodes_by_name(self, name):
        result = [k for k, v in self.nodes(data=True) if v.get("name") == name]
        return result

    def subgraph_from(self, node):
        from collections import deque

        Q = deque([node])
        found = set()
        while len(Q):
            i = Q.popleft()
            if i not in found:
                found.add(i)
                Q.extend(self.successors(i))
        return NestingTree(self.subgraph(found), root_id=node)

    def stats_summarize(self):
        print("Graph Stats")
        print(f"  Overall: {len(self)} nodes")
        tier = [self.root_id]
        next_tier = list(self.successors(self.root_id))
        n = 0
        while len(next_tier):
            tier = next_tier
            n += 1
            print(f"   Tier {n}: {len(tier)} nodes")
            next_tier = list()
            for i in tier:
                next_tier.extend(self.successors(i))

    def node_slot_arrays(self, model, parameter_dict=None):
        if hasattr(model, "get_slot_x"):
            muslots = np.full([len(self)], -1, dtype=np.int32)
            for child, childcode in enumerate(self.standard_sort):
                # for parent in self.predecessor_slots(childcode):
                # 	alpha[parent, child] = 1
                pname = self.nodes[childcode].get("parameter", None)
                muslots[child] = model.get_slot_x(pname)
        else:
            muslots = np.ones([len(self)], dtype=model)
            for child, childcode in enumerate(self.standard_sort):
                # for parent in self.predecessor_slots(childcode):
                # 	alpha[parent, child] = 1
                pname = self.nodes[childcode].get("parameter", None)
                if pname is not None:
                    if parameter_dict is not None and isinstance(pname, str):
                        pname = parameter_dict.get(pname, pname)
                    muslots[child] = model(pname)
        num = np.zeros(len(self.nodes), dtype=np.int32)
        start = np.full(len(self.nodes), -1, dtype=np.int32)
        n = self.n_edges
        for upcode in reversed(self.standard_sort):
            upslot = self.standard_slot_map[upcode]
            for dnslot in reversed(self.successor_slots(upcode)):
                n -= 1
                num[upslot] += 1
                start[upslot] = n
        return (
            muslots,
            start,
            num,
        )

    def _get_simple_mu_and_alpha(self, model, holdfast_invalidates=True):
        # alpha = np.zeros([len(self), len(self)], dtype=np.float64)
        mu = np.ones(
            [
                len(self),
            ],
            dtype=np.float64,
        )
        muslots = np.full(
            [
                len(self),
            ],
            -1,
            dtype=np.int32,
        )
        for child, childcode in enumerate(self.standard_sort):
            # for parent in self.predecessor_slots(childcode):
            # 	alpha[parent, child] = 1
            pname = self.nodes[childcode].get("parameter", None)
            mu[child] = model.get_value(pname, default=1.0)
            muslots[child] = model.get_slot_x(pname, holdfast_invalidates)

        s = self.n_edges
        up = np.zeros(s, dtype=np.int32)
        dn = np.zeros(s, dtype=np.int32)
        val = np.zeros(s, dtype=np.float64)
        num = np.zeros(len(self.nodes), dtype=np.int32)
        start = np.full(len(self.nodes), -1, dtype=np.int32)
        # first_visit = np.zeros(s, dtype=np.int32)
        n = s
        # first_visit_found = set()
        for upcode in reversed(self.standard_sort):
            upslot = self.standard_slot_map[upcode]
            for dnslot in reversed(self.successor_slots(upcode)):
                n -= 1
                up[n] = upslot
                dn[n] = dnslot
                num[upslot] += 1
                start[upslot] = n
                val[n] = 1 / len(
                    self.predecessor_slots(self.standard_sort[dnslot])
                )  # TODO make not always constant fraction
        # for n in range(s):
        # 	if dn[n] not in first_visit_found:
        # 		first_visit[n] = 1
        # 		first_visit_found.add(dn[n])

        return mu, muslots, up, dn, num, start, val

    def as_arrays(self, model=np.float32, trim=False, parameter_dict=None):
        """
        Express this tree as a dict of arrays for use with sharrow.

        Parameters
        ----------
        model : Model or dtype
            Give a model to extract MU values as parameter slot positions,
            or a dtype to extract as
        trim : bool, default False
            Trim the node slot arrays to be only for nests.
        parameter_dict : Mapping[str,Number], optional
            Maps named parameters to values.

        Returns
        -------
        dict
        """
        result = {}
        result["n_nodes"] = len(self)
        result["n_alts"] = n_alts = self.n_elementals()
        up, dn, first_visit, alloc_slot = self.edge_slot_arrays()
        result["edges_up"] = up
        result["edges_dn"] = dn
        result["edges_1st"] = first_visit
        result["edges_alloc"] = alloc_slot
        muslots, start, num = self.node_slot_arrays(
            model=model, parameter_dict=parameter_dict
        )
        if trim:
            muslots = muslots[n_alts:]
            start = start[n_alts:]
            num = num[n_alts:]
        result["mu_params"] = muslots
        result["start_slots"] = start
        result["len_slots"] = num
        return result


def graph_to_figure(graph, output_format="svg", **format):

    try:
        import pygraphviz as viz
    except ImportError:
        import warnings

        warnings.warn("pygraphviz module not installed, unable to draw nesting tree")
        raise NotImplementedError(
            "pygraphviz module not installed, unable to draw nesting tree"
        )
    existing_format_keys = list(format.keys())
    for key in existing_format_keys:
        if key.upper() != key:
            format[key.upper()] = format[key]
    if "SUPPRESSGRAPHSIZE" not in format:
        if "GRAPHWIDTH" not in format:
            format["GRAPHWIDTH"] = 6.5
        if "GRAPHHEIGHT" not in format:
            format["GRAPHHEIGHT"] = 4
    if "UNAVAILABLE" not in format:
        format["UNAVAILABLE"] = True
    # x = XML_Builder("div", {'class':"nesting_graph larch_art"})
    # x.h2("Nesting Structure", anchor=1, attrib={'class':'larch_art_xhtml'})
    from io import BytesIO

    if "SUPPRESSGRAPHSIZE" not in format:
        G = viz.AGraph(
            name="Tree",
            directed=True,
            size="{GRAPHWIDTH},{GRAPHHEIGHT}".format(**format),
        )
    else:
        G = viz.AGraph(name="Tree", directed=True)
    for n in graph.nodes:
        nname = graph.nodes[n].get("name", n)
        if nname == n:
            G.add_node(
                n, label="<{1}>".format(n, nname), style="rounded,solid", shape="box"
            )
        else:
            G.add_node(
                n,
                label='<{1} <FONT COLOR="#999999">({0})</FONT>>'.format(n, nname),
                style="rounded,solid",
                shape="box",
            )
    try:
        graph.elementals
    except AttributeError:
        pass
    else:
        eG = G.add_subgraph(
            name="cluster_elemental",
            nbunch=graph.elementals,
            color="#cccccc",
            bgcolor="#eeeeee",
            label="Elemental Alternatives",
            labelloc="b",
            style="rounded,solid",
        )
    unavailable_nodes = set()
    # if format['UNAVAILABLE']:
    # 	if self.is_provisioned():
    # 		try:
    # 			for n, ncode in enumerate(self.alternative_codes()):
    # 				if np.sum(self.Data('Avail'),axis=0)[n,0]==0: unavailable_nodes.add(ncode)
    # 		except: raise
    # 	try:
    # 		legible_avail = not isinstance(self.df.queries.avail, str)
    # 	except:
    # 		legible_avail = False
    # 	if legible_avail:
    # 		for ncode,navail in self.df.queries.avail.items():
    # 			try:
    # 				if navail=='0': unavailable_nodes.add(ncode)
    # 			except: raise
    # 	eG.add_subgraph(name='cluster_elemental_unavailable', nbunch=unavailable_nodes, color='#bbbbbb', bgcolor='#dddddd',
    # 				   label='Unavailable Alternatives', labelloc='b', style='rounded,solid')
    try:
        G.add_node(graph.root_id, label="Root")
    except AttributeError:
        pass
    up_nodes = set()
    down_nodes = set()
    for i, j in graph.edges:
        G.add_edge(i, j)
        down_nodes.add(j)
        up_nodes.add(i)
    pyg_imgdata = BytesIO()
    try:
        G.draw(
            pyg_imgdata, format=output_format, prog="dot"
        )  # write postscript in k5.ps with neato layout
    except ValueError as err:
        if "in path" in str(err):
            import warnings

            warnings.warn(str(err) + "; unable to draw nesting tree in report")
            raise NotImplementedError()
    from xmle import Elem

    if output_format == "svg":
        import xml.etree.ElementTree as ET

        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        result = ET.fromstring(pyg_imgdata.getvalue().decode())
    else:
        result = Elem(
            "span",
            attrib={"style": "color:red"},
            text=f"Unable to render output_format '{output_format}'",
        )
    x = Elem("div") << result
    return x


def reverse_lexicographical_topological_sort(G, key=None):
    """
    Generator of nodes in reverse lexicographically topologically sorted order.

    A general topological sort is a nonunique permutation of the nodes such that
    an edge from u to v implies that u appears before v in the topological sort
    order.

    The lexicographical topological sort breaks ties by ordering according to
    node labels, so that the sorting becomes unique.

    Parameters
    ----------
    G : NetworkX digraph
            A directed acyclic graph (DAG)

    key : function, optional
            This function maps nodes to keys with which to resolve ambiguities in
            the sort order.  Defaults to the identity function.

    Returns
    -------
    iterable
            An iterable of node names in lexicographical topological sort order.

    Raises
    ------
    NetworkXError
            Topological sort is defined for directed graphs only. If the graph `G`
            is undirected, a :exc:`NetworkXError` is raised.

    NetworkXUnfeasible
            If `G` is not a directed acyclic graph (DAG) no topological sort exists
            and a :exc:`NetworkXUnfeasible` exception is raised.  This can also be
            raised if `G` is changed while the returned iterator is being processed

    RuntimeError
            If `G` is changed while the returned iterator is being processed.

    """

    if not G.is_directed():
        msg = "Topological sort not defined on undirected graphs."
        raise nx.NetworkXError(msg)

    if key is None:

        def key(node):
            return node

    nodeid_map = {n: i for i, n in enumerate(G)}

    def create_tuple(node):
        return key(node), nodeid_map[node], node

    outdegree_map = {v: d for v, d in G.out_degree() if d > 0}
    # These nodes have zero outdegree and ready to be returned.
    zero_outdegree = [create_tuple(v) for v, d in G.out_degree() if d == 0]
    heapq.heapify(zero_outdegree)

    while zero_outdegree:
        _, _, node = heapq.heappop(zero_outdegree)

        if node not in G:
            raise RuntimeError("Graph changed during iteration")
        for parent, child in G.in_edges(node):
            try:
                outdegree_map[parent] -= 1
            except KeyError as e:
                raise RuntimeError("Graph changed during iteration") from e
            if outdegree_map[parent] == 0:
                heapq.heappush(zero_outdegree, create_tuple(parent))
                del outdegree_map[parent]

        yield node

    if zero_outdegree:
        msg = "Graph contains a cycle or graph changed during iteration"
        raise nx.NetworkXUnfeasible(msg)
