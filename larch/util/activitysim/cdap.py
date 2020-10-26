import numpy as np
import pandas as pd
import re
import itertools
from larch import P, X, DataFrames, Model
from larch.model.model_group import ModelGroup
from larch.util import Dict

import logging
from larch.log import logger_name

_logger = logging.getLogger(logger_name)


def generate_alternatives(n_persons):
    """
    Generate a dictionary of CDAP alternatives.

    The keys are the names of the patterns, and
    the values are the alternative code numbers.

    Parameters
    ----------
    n_persons : int

    Returns
    -------
    Dict
    """
    basic_patterns = ["M", "N", "H"]
    alt_names = list(
        "".join(i) for i in itertools.product(basic_patterns, repeat=n_persons)
    )
    alt_codes = np.arange(1, len(alt_names) + 1)
    return Dict(zip(alt_names, alt_codes))


def apply_replacements(expression, prefix, tokens):
    """
    Convert general person terms to specific person terms for the CDAP model.

    Parameters
    ----------
    expression : str
        An expression from the "Expression" column
        of cdap_INDIV_AND_HHSIZE1_SPEC.csv, or similar.
    prefix : str
        A prefix to attach to each token in `expression`.
    tokens : list-like of str
        A list of tokens to edit within an expression.

    Returns
    -------
    expression : str
        The modified expression
    """
    for i in tokens:
        expression = re.sub(fr"\b{i}\b", f"{prefix}_{i}", expression)
    return expression


def cdap_base_utility_by_person(model, n_persons, spec, alts=None, value_tokens=()):
    """
    Build the base utility by person for each pattern.

    Parameters
    ----------
    model : larch.Model
    n_persons : int
    spec : pandas.DataFrame
        The base utility by person spec provided by
        the ActivitySim framework.
    alts : dict, optional
        The keys are the names of the patterns, and
        the values are the alternative code numbers,
        as created by `generate_alternatives`.  If not
        given, the alts are automatically regenerated
        using that function.
    value_tokens : list-like of str, optional
        A list of tokens to edit within an the expressions,
        generally the column names of the provided values
        from the estimation data bundle.  Only used when
        `n_persons` is more than 1.
    """
    if n_persons == 1:
        for i in spec.index:
            if not pd.isna(spec.loc[i, "M"]):
                model.utility_co[1] += X(spec.Expression[i]) * P(f"coef_{i:03d}_M")
            if not pd.isna(spec.loc[i, "N"]):
                model.utility_co[2] += X(spec.Expression[i]) * P(f"coef_{i:03d}_N")
            if not pd.isna(spec.loc[i, "H"]):
                model.utility_co[3] += X(spec.Expression[i]) * P(f"coef_{i:03d}_H")
    else:
        if alts is None:
            alts = generate_alternatives(n_persons)
        person_numbers = range(1, n_persons + 1)
        for pnum in person_numbers:
            for i in spec.index:
                for aname, anum in alts.items():
                    z = pnum - 1
                    if not pd.isna(spec.loc[i, aname[z]]):
                        x = apply_replacements(
                            spec.Expression[i], f"p{pnum}", value_tokens
                        )
                        model.utility_co[anum] += X(x) * P(f"coef_{i:03d}_{aname[z]}")


def interact_pattern(n_persons, select_persons, tag):
    """
    Compile a regex pattern to match CDAP alternatives.

    Parameters
    ----------
    n_persons : int
    select_persons : list-like of int
        The persons to be selected.
    tag : str
        The activity letter, currently one of {M,N,H}.

    Returns
    -------
    re.compile
    """

    pattern = ""
    p = 1
    while len(pattern) < n_persons:
        pattern += tag if p in select_persons else "."
        p += 1
    return re.compile(pattern)


def cdap_interaction_utility(model, n_persons, alts, interaction_coef):

    person_numbers = list(range(1, n_persons + 1))

    for (cardinality, activity), coefs in interaction_coef.groupby(
        ["cardinality", "activity"]
    ):
        _logger.info(f"{n_persons} person households, interaction cardinality {cardinality}, activity {activity}")
        if cardinality > n_persons:
            continue
        elif cardinality == n_persons:
            this_aname = activity * n_persons
            this_altnum = alts[this_aname]
            for rowindex, row in coefs.iterrows():
                expression = "&".join(
                    f"(p{p}_ptype == {t})"
                    for (p, t) in zip(person_numbers, row.interaction_ptypes)
                )
                if expression:
                    linear_component = X(expression) * P(f"interaction_{rowindex:03d}_{row.slug}")
                else:
                    linear_component = P(f"interaction_{rowindex:03d}_{row.slug}")
                _logger.debug(
                    f"utility_co[{this_altnum} {this_aname}] += {linear_component}"
                )
                model.utility_co[this_altnum] += linear_component
        elif cardinality < n_persons:
            for combo in itertools.combinations(person_numbers, cardinality):
                pattern = interact_pattern(n_persons, combo, activity)
                for aname, anum in alts.items():
                    if pattern.match(aname):
                        for rowindex, row in coefs.iterrows():
                            expression = "&".join(
                                f"(p{p}_ptype == {t})"
                                for (p, t) in zip(combo, row.interaction_ptypes)
                            )
                            # interaction terms without ptypes (i.e. with wildcards)
                            # only apply when the household size matches the cardinality
                            if expression != "":
                                linear_component = X(expression) * P(
                                    f"interaction_{rowindex:03d}_{row.slug}"
                                )
                                _logger.debug(
                                    f"utility_co[{anum} {aname}] += {linear_component}"
                                )
                                model.utility_co[anum] += linear_component


def cdap_split_data(households, values):
    if "cdap_rank" not in values:
        raise ValueError("assign cdap_rank to values first")
    # only process the first 5 household members
    values = values[values.cdap_rank <= 5]
    cdap_data = {}
    for hhsize, hhs_part in households.groupby(households.hhsize.clip(1, 5)):
        if hhsize == 1:
            v = pd.merge(values, hhs_part.household_id, on="household_id").set_index(
                "household_id"
            )
        else:
            v = (
                pd.merge(values, hhs_part.household_id, on="household_id")
                .set_index(["household_id", "cdap_rank"])
                .unstack()
            )
            v.columns = [f"p{i[1]}_{i[0]}" for i in v.columns]
            for agglom in ["override_choice", "model_choice"]:
                v[agglom] = v[[f"p{p}_{agglom}" for p in range(1, hhsize + 1)]].sum(1)
        cdap_data[hhsize] = v
    return cdap_data


def cdap_dataframes(households, values):
    data = cdap_split_data(households, values)
    dfs = {}
    for hhsize in data.keys():
        alts = generate_alternatives(hhsize)
        dfs[hhsize] = DataFrames(
            co=data[hhsize],
            alt_names=alts.keys(),
            alt_codes=alts.values(),
            av=1,
            ch=data[hhsize].override_choice.map(alts),
        )
    return dfs


def cdap_model(households, values, spec1, interaction_coef):
    cdap_data = cdap_dataframes(households, values)
    m = {}
    _logger.info(f"building for model 1")
    m[1] = Model(dataservice=cdap_data[1])
    cdap_base_utility_by_person(m[1], n_persons=1, spec=spec1)
    m[1].choice_any = True
    m[1].availability_any = True

    for s in [2, 3, 4, 5]:
        _logger.info(f"building for model {s}")
        m[s] = Model(dataservice=cdap_data[s])
        alts = generate_alternatives(s)
        cdap_base_utility_by_person(m[s], s, spec1, alts, values.columns)
        cdap_interaction_utility(m[s], s, alts, interaction_coef)
        m[s].choice_any = True
        m[s].availability_any = True

    return ModelGroup(m.values())
