from collections import Counter
from itertools import product
from typing import Sequence
import numpy as np
from sympy import symbols, Symbol
from sympy.logic.boolalg import to_cnf, And, Xor, Not, Or
from sympy.logic.boolalg import Equivalent as Eq
from sympy.logic.boolalg import Implies as Imp

def flatten(sequence, dtype=Sequence):
    for item in sequence:
        if isinstance(item, dtype):
            yield from flatten(item)
        else:
            yield item

def concentration(courses: Sequence[dict], n: int = 7):
    ranked = sorted([c for c in courses if c['type'] in ('HUM', 'SSC', 'SCI')], key = lambda c: c['rank'])
    top = ranked[-n:]
    counts = Counter(c['type'] for c in top)
    return max(counts, key = lambda c: counts[c])

def make_time_segments(y: int = 3, p: int = 6):
    ys = tuple(range(1, y + 1))
    ps = tuple(range(1, p + 1))
    return tuple(product(ps, ys))

def make_edges(courses: Sequence[dict], time_segments: Sequence[tuple[int, int]]):
    #NOTE: For the demo, we remove the edges assigning Capstone to the first semester of final year or earlier.
    _, years = zip(*time_segments)
    max_y = max(years)

    edges = []
    edge_to_course = dict()
    for c in courses:
        for t in time_segments:
            p, y = t
            if p in flatten(c['period']):
                e = (c['code'], p, y)

                if e[0].startswith('CAP') and (p <= 3 or y < max_y):
                    continue

                edges.append(e)
                edge_to_course[e] = c

    return tuple(edges), edge_to_course

def make_alloc_vars(edges: tuple[tuple[str, int, int]]):
    nametag = {(c, p, y): f'{c}OnPeriod{p}Year{y}' for (c, p, y) in edges}
    edge_to_alloc = {e: symbols(nametag[e]) for e in edges}
    alloc_to_edge = {v: e for (e, v) in edge_to_alloc.items()}
    return edge_to_alloc, alloc_to_edge

def make_state_vars(courses: Sequence[dict]):
    return {c['code']: symbols(f'{c["code"]}isAlloc') for c in courses}

def make_course_to_edge(edges: tuple[tuple[str, int, int]]):
    ctoe = dict()
    for e in edges:
        c, p, y = e
        if c not in ctoe:
            ctoe[c] = []
        ctoe[c].append(e)
    return ctoe

def is_core(course: dict):
    return course['code'] in ('COR1002', 'COR1003', 'COR1004', 'COR1006', 'SKI1008', 'SKI1009', 'SKI1004', 'SKI1005', 'PRO1010', 'PRO1012')

def is_project(course: dict):
    return course['type'] in ('PRO', 'UGR')

def is_skill(course: dict):
    return course['type'] in ('SKI')

def is_capstone(course: dict):
    return course['type'] in ('CAP')

def is_elective(course: dict):
    return not is_core(course) and not is_skill(course) and not is_project(course) and not is_capstone(course)

def work_load(course: dict):
    avg_len = sum(len(t) for t in course['period']) / len(course['period'])
    return course['ects'] / avg_len

def _make_course_time_CNF(course: dict, years: Sequence[int], edge_to_alloc: dict, state_vars: dict):
    xor_op = []
    for y in years:
        for periods in course['period']:
           vset = [edge_to_alloc[(course['code'], p, y)] for p in periods if (course['code'], p, y) in edge_to_alloc]
           if vset:
                and_op = And(*vset)
                xor_op.append(and_op)
    formula = Eq(state_vars[course['code']], Or(*xor_op))
    return to_cnf(formula)

def make_time_CNF(courses: Sequence[dict], time_segments: Sequence[tuple[int, int]], edge_to_alloc: dict, state_vars: dict):
    _, ys = zip(*time_segments)
    years = frozenset(ys)
    return And(*[_make_course_time_CNF(c, years, edge_to_alloc, state_vars) for c in courses])

def or_to_ineq(or_op):

    if isinstance(or_op, Symbol):
        return 1., {a: 1. for a in or_op.atoms()}, np.inf
    
    if isinstance(or_op, Not):
        return 0., {a: -1. for a in or_op.atoms()}, np.inf
    
    if isinstance(or_op, Or):
        coeff = dict()
        num = 0
        for prop in or_op.args:
            if isinstance(prop, Not):
                v = prop.args[0]
                coeff[v] = -1.
                num += 1

            else:
                v = prop
                coeff[v] = 1.

        return 1. - num, coeff, np.inf

    raise TypeError(f'or_op should not be {type(or_op)}')

def CNF_to_ineq(cnf):
    return [or_to_ineq(or_op) for or_op in cnf.args]

def make_time_ineq(courses: Sequence[dict], time_segments: Sequence[tuple[int, int]], edge_to_alloc: dict, state_vars: dict):
    return CNF_to_ineq(make_time_CNF(courses, time_segments, edge_to_alloc, state_vars))

def make_core_ineq(courses: Sequence[dict], state_vars):
    return [(10., {state_vars[c['code']]: 1. for c in courses if is_core(c)}, 10.)]

def make_gedu_ineq(courses: Sequence[dict], state_vars: dict):
    conc = concentration(courses)
    return [(2., {
        state_vars[c['code']]: 1. for c in courses if c['type'] in ('HUM', 'SCI', 'SSC') and c['type'] != conc
        }, 4.)]

def make_conc_ineq(courses: Sequence[dict], state_vars: dict):
    conc = concentration(courses)
    ineq = []
    
    ineq.append((14., {state_vars[c['code']]: 1. for c in courses if is_elective(c) and c['type'] == conc}, 16.))
    ineq.append((0., {state_vars[c['code']]: 1. for c in courses if is_elective(c) and c['type'] == conc and c['level'] == 1}, 4.))
    ineq.append((4., {state_vars[c['code']]: 1. for c in courses if is_elective(c) and c['type'] == conc and c['level'] == 3}, np.inf))

    ineq.append((6., {state_vars[c['code']]: 1. for c in courses if is_skill(c) and not is_core(c) and c['level'] in (2, 3)}, 6.))

    ineq.append((3., {state_vars[c['code']]: 1. for c in courses if is_project(c) and not is_core(c)}, 3.))
    ineq.append((0., {state_vars[c['code']]: 1. for c in courses if is_project(c) and not is_core(c) and c['level'] == 2}, 2.))
    ineq.append((1., {state_vars[c['code']]: 1. for c in courses if is_project(c) and not is_core(c) and c['level'] == 3}, np.inf))

    return ineq

def make_caps_ineq(courses: Sequence[dict], state_vars: dict):
    ineqs = [(1., {state_vars[c['code']]: 1. for c in courses if is_capstone(c)}, 1.)]
    return ineqs

def make_ects_ineq(courses: Sequence[dict], state_vars: dict):
    return [(180., {state_vars[c['code']]: c['ects'] for c in courses}, 180.)]

def make_load_ineq(edges: tuple[tuple[str, int, int]], time_segments: Sequence[tuple[int, int]], edge_to_alloc: dict, edge_to_course: dict):
    ineq = []
    ps, ys = zip(*time_segments)
    years = frozenset(ys)
    for y in years:
        for p in ps:
            ineq.append(
                (0., {edge_to_alloc[(c, _p, _y)]: work_load(edge_to_course[(c, _p, _y)]) for (c, _p, _y) in edges if p == _p and y == _y}, 15.)
                )
    return ineq

def ineq_to_lb_A_ub(ineq, idxmap):
    row = np.zeros((len(idxmap) + 2,))
    
    for v, w in ineq[1].items():
        row[idxmap[v] + 1] = w

    row[0] = ineq[0]
    row[-1] = ineq[-1]
    return row

from scipy.sparse import dok_array

def all_ineq_to_lb_A_ub(ineq_list, idxmap):
    rows = []
    for ineq in ineq_list:
        rows.append(ineq_to_lb_A_ub(ineq, idxmap))

    M = np.stack(rows, axis=0)
    return M[:, 0], M[:, 1:-1], M[:, -1]

from typing import Sequence
from itertools import combinations
from sympy.logic.boolalg import to_cnf, And, Implies, Not, Or
from sympy import symbols
import pyparsing as pp
import json

def make_parser():
    code = pp.Combine(pp.Word(pp.alphas) + pp.Word(pp.nums))
    func = pp.Group(pp.Word(pp.alphas) + pp.Suppress('(') + pp.Word(pp.nums) + pp.Suppress(',') + pp.Word(pp.nums) + pp.Suppress(')'))
    atom = code | func
    expr = pp.infix_notation(atom, [("NOT", 1, pp.opAssoc.RIGHT), ('OR', 2, pp.opAssoc.LEFT), ('AND', 2, pp.OpAssoc.LEFT)], lpar='(', rpar=')')
    return expr

def to_int(T: Sequence[str | Sequence] | str):
    if isinstance(T, str) and T.isdigit():
        return int(T)
    
    elif (not isinstance(T, str)) and isinstance(T, Sequence) and len(T) > 0:
        return [to_int(t) for t in T]
    
    return T

def is_func(T: Sequence[str | int | Sequence]):
    if isinstance(T, Sequence) and len(T) == 3 and isinstance(T[0], str) and isinstance(T[1], int) and isinstance(T[2], int):
        return True
    
    return False

def unroll_func(T: Sequence[str | int | Sequence], courses: Sequence[dict]):
    if isinstance(T, str):
        return T

    if is_func(T):
        typ = T[0]
        num = T[1]
        lvl = T[2]
        C = {c['code'] for c in courses if c['type'] == typ and c['level'] >= lvl}
        
        if not C:
            return []
        
        return ['OR', *list(combinations(C, num))]

    return [unroll_func(t, courses) for t in T]

def to_symbols(T: Sequence[Sequence | str] | str, sym: dict):
    if isinstance(T, str):
        assert T in sym, f'{T} is not registered as a sympy symbol'
        return sym[T]
    
    if isinstance(T, Sequence) and len(T) == 0:
        return None
    
    opr = Or if 'OR' in T else (Not if 'NOT' in T else And)
    lofs = [to_symbols(t, sym) for t in T if t not in ('NOT', 'AND', 'OR')]
    return opr(*[lof for lof in lofs if lof is not None])

def parse(formula: str, parser: pp.ParserElement, courses: Sequence[dict], sym: dict):
    res = parser.parse_string(formula, parseAll=True).as_list()
    res = to_int(res)
    res = unroll_func(res, courses)
    res = to_symbols(res, sym)
    return res

def prereq_mapping(courses: Sequence[dict]):
    parser = make_parser()
    sym = {c['code']: symbols(c['code']) for c in courses}

    res = dict()
    for c in courses:
        if len(c['prereq_']) > 0:
            res[sym[c['code']]] = parse(c['prereq_'], parser, courses, sym)
    
    return res

def unroll_mapping(mapping, edges, edge_to_alloc, as_cnf: bool = True):
    unrolled = []
    for condition, consequence in mapping.items():
        left_edges = tuple((c, p, y) for (c, p, y) in edges if c == str(condition))
        assert left_edges

        for le in left_edges:
            subs = dict()
            for a in consequence.atoms():
                if a not in condition.atoms():
                    chain = []
                    for (c, p, y) in edges:
                        if c == str(a) and (y < le[2] or (p < le[1] and y == le[2])):
                            chain.append(edge_to_alloc[(c, p, y)])
                    subs[a] = Or(*chain)
                else:
                    subs[a] = False
            
            formula = Implies(edge_to_alloc[le], consequence.subs(subs))
            unrolled.append(to_cnf(formula) if as_cnf else formula)

    return unrolled

def impls_to_ineq(impls):
    return CNF_to_ineq(And(*impls))

def make_prereq_ineq(courses, edges, edge_to_alloc):
    mapping = prereq_mapping(courses)
    impls = unroll_mapping(mapping, edges, edge_to_alloc)
    return impls_to_ineq(impls)

import time
import numpy as np
from itertools import product
from scipy.optimize import milp, LinearConstraint
from typing import Callable

from copy import deepcopy

def norm_periods(courses: Sequence[dict]):
    for course in courses:
        if isinstance(course["period"], list) and course["period"]:
            for i, item in enumerate(course["period"]):
                if isinstance(item, int):
                    course["period"][i] = [item]

    return courses

class UCMPlanner:
    def __init__(self, reclib: Sequence[dict] | str) -> None:

        if isinstance(reclib, str):
            with open(reclib, 'r') as file:
                self.courses = json.load(file)

        else:
            self.courses = deepcopy(reclib)

        self.courses = norm_periods(self.courses)
        self.code_to_course = {c['code']: c for c in self.courses}

        for c in self.courses:
            c['rank'] = 0.

        self.tseg = make_time_segments()
        self.edges, self.edge_to_course = make_edges(self.courses, self.tseg)
        self.course_to_edge = make_course_to_edge(self.edges)

        self.edge_to_alloc, self.alloc_to_edge = make_alloc_vars(self.edges)
        self.state_vars = make_state_vars(self.courses)

        time_ineq = make_time_ineq(self.courses, self.tseg, self.edge_to_alloc, self.state_vars)
        core_ineq = make_core_ineq(self.courses, self.state_vars)
        '''gedu_ineq = make_gedu_ineq(self.courses, self.state_vars)
        conc_ineq = make_conc_ineq(self.courses, self.state_vars)'''
        caps_ineq = make_caps_ineq(self.courses, self.state_vars)
        ects_ineq = make_ects_ineq(self.courses, self.state_vars)
        load_ineq = make_load_ineq(self.edges, self.tseg, self.edge_to_alloc, self.edge_to_course)
        prereq_ineq = make_prereq_ineq(self.courses, self.edges, self.edge_to_alloc)
        self.ineq_list = time_ineq + load_ineq + core_ineq + caps_ineq + ects_ineq + prereq_ineq

        self.alloc_vars_list = [v for v in self.edge_to_alloc.values()]
        self.state_vars_list = [v for v in self.state_vars.values()]
        self.all_vars_list = self.alloc_vars_list + self.state_vars_list
        self.idxmap = {v: i for i, v in enumerate(self.all_vars_list)}

    def _update_ranks(self, courses: Sequence[dict]):
        for c in courses:
            ref = self.code_to_course[c['code']]
            ref['rank'] = c['score'] if 'score' in c else c['rank']

    def plan(self, courses: Sequence[dict]):
        self._update_ranks(courses)

        gedu_ineq = make_gedu_ineq(self.courses, self.state_vars)
        conc_ineq = make_conc_ineq(self.courses, self.state_vars)

        lb, A, ub = all_ineq_to_lb_A_ub(self.ineq_list + gedu_ineq + conc_ineq, self.idxmap)
        lb_A_ub = LinearConstraint(A, lb, ub)
        
        coeff = np.zeros((A.shape[1],))
        for v in self.alloc_vars_list:
            i = self.idxmap[v]
            e = self.alloc_to_edge[v]
            w = self.edge_to_course[e]['rank']
            coeff[i] = w

        res = milp(c=-coeff, integrality=[1] * A.shape[1], bounds=(0., 1.), constraints=lb_A_ub)

        if res.x is not None:
            ps, ys = zip(*self.tseg)
            years = frozenset(ys)
            periods = frozenset(ps)
            tree = {
                y: {p: [] for p in periods} for y in years
            }

            for course in self.courses:
                es = self.course_to_edge[course['code']]
                for e in es:
                    v = self.edge_to_alloc[e]
                    i = self.idxmap[v]

                    if res.x[i] > 0.5:
                        _, p, y = e
                        tree[y][p].append({'code': course['code'], 'title': course['title']})
                    
            pretty_tree = {
                f'Year_{y}': {f'Period_{p}': tree[y][p] for p in periods} for y in years
            }

            return pretty_tree, res

        return None, res