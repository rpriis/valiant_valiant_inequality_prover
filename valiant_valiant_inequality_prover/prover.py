import numpy as np
from itertools import combinations
from scipy.linalg import null_space
from scipy.optimize import linprog

SEQS_MIN = 1
EXPRS_MIN = 1
ZERO_BOUND = 1e-8


def get_combs(n, length):
    return list(combinations(range(n), length + 1))


def is_zero_ish(x):
    return abs(x) < ZERO_BOUND


def values_close(a, b):
    return abs(a - b) < ZERO_BOUND


rows_close_by_line = np.frompyfunc(values_close, 2, 1)

vector_element_is_zero_ish = np.frompyfunc(is_zero_ish, 1, 1)


def rows_close(u, v):
    return np.all(rows_close_by_line(u, v))


def merge_common_matrix_expressions(m):
    if m is None:
        return
    if m.shape[1] == 0:
        raise ValueError("Invalid input matrix: lacks sufficient dimension")
    if m.shape[0] == 0:
        return m

    m_non_zero = m[np.where(m[:, -1])[0]]
    if m_non_zero.shape[0] < 2:  # not enough rows to bother merging
        return m_non_zero

    num_rows = m_non_zero.shape[0]
    included_row_indices = []
    i = 0
    while i < num_rows:
        row_is_unique = True
        if is_zero_ish(m_non_zero[i, -1], ):
            continue
        for j in included_row_indices:
            if rows_close(m_non_zero[j, :-1], m_non_zero[i, :-1]):
                m_non_zero[j, -1] += m_non_zero[i, -1]
                row_is_unique = False
                break
        if row_is_unique:
            included_row_indices.append(i)
        i += 1

    # Filter to merged rows
    proposed_m = m_non_zero[sorted(included_row_indices)]
    # remove rows with near-zero powers.
    proposed_m_non_zero = proposed_m[np.where(1 - vector_element_is_zero_ish(proposed_m[:, -1]))[0]]

    return proposed_m_non_zero


def get_latex(m, mode='inline', include_geq=True, simplify=False):
    if include_geq != 0 and include_geq != 1:
        raise ValueError("Invalid value for include_geq. Expected True or False")
    if mode not in ['full', 'inline', 'raw']:
        raise ValueError("Invalid mode. Expected one of: %s" % mode)
    ornaments = {'full': ('\\begin{equation*}\n\t', '\n\\end{equation*}'), 'inline': ('$', '$'), 'raw': ('', '')}

    if m.shape[0] == 0:
        return ornaments[mode][0] + r'1\geq1' + ornaments[mode][1]

    if m.shape[0] < SEQS_MIN or m.shape[1] < EXPRS_MIN + 1:
        raise ValueError(
            "Input matrix is of insufficient size. Expected at least %d sequences and %d expressions %s" % SEQS_MIN,
            EXPRS_MIN)

    latex_out = ornaments[mode][0]
    for i in range(m.shape[0]):
        latex_out += r'\left(\sum_j'
        if np.all(vector_element_is_zero_ish(m[i, :-1])):
            latex_out += '1'
        for j in range(m.shape[1] - 1):
            if is_zero_ish(m[i, j]) and simplify:  # power is 0
                continue
            latex_out += f"{chr(ord('a') + j)}_j"
            if not values_close(m[i, j], 1) or not simplify:  # power isn't 1
                latex_out += '^{' + '%3.3g' % m[i, j] + '}'
        latex_out += r'\right)^{' + '%3.3g' % m[i, -1] + '}'
    latex_out += r'\geq1' * include_geq + ornaments[mode][1]

    return latex_out


def matrix_round_near_zeroes(m):
    m_flat = m.ravel()
    return np.where(vector_element_is_zero_ish(m_flat), 0, m_flat).reshape(m.shape)


class SolverResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# 'merge_expressions' merges expressions before constructing the linear program
def ineq_solver(inA, merge_common_expressions=True, round_near_zeroes=True):
    """
    Decompose given inequality into Holder and Lp-monotonicity inequalities or asserts its invalidity

    Parameters
    ----------
    inA : 2D numpy.matrix with datatype numpy.float64
        The input expressions, composed of rows, that form the proposed inequality
    merge_common_expressions : bool
        If equivalent expressions are merged before processing, condensing the returned result
    round_near_zeroes : bool
        If near-zero values should be rounded before returning

    Return
    ------
    dict of:
        eq_valid : bool
            Whether the proposed inequality is valid
        holder_ineqs_matrix : numpy.matrix with datatype numpy.float64
            Holder inequalities contained in the decomposition of proposed inequality
        holder_ineqs_overall_powers : numpy.array with datatype numpy.float64
            Powers of inequalities w.r.t. holder_ineqs_matrix
        lp_mon_ineqs_matrix : numpy.matrix with datatype numpy.float64
            Lp-monotonicity inequalities contained in the decomposition of proposed inequality
        lp_mon_ineqs_overall_powers : numpy.array with datatype numpy.float64
            Powers of inequalities w.r.t. lp_mon_ineqs_matrix

    Examples
    --------
    >>> import numpy as np
    >>> inA = np.matrix([[1, 0, 0.5], [0, 1, 0.5], [0.5, 0.5, -1]], dtype=np.float64)
    >>> ineq_result = ineq_solver(inA)
    >>> ineq_result.eq_valid
    True
    >>> ineq_result.holder_ineqs_matrix
    array([[[ 1. ,  0. ,  0.5],
            [ 0. ,  1. ,  0.5],
            [ 0.5,  0.5, -1. ]]])
    >>> ineq_result.holder_ineqs_overall_powers
    array([1.])
    >>> ineq_result.lp_mon_ineqs_matrix
    array([], shape=(0, 2, 3), dtype=float64)
    >>> ineq_result.lp_mon_ineqs_overall_powers
    array([], dtype=float64)
    """
    if merge_common_expressions:
        inA = merge_common_matrix_expressions(inA)

    r = inA.shape[0]
    dim = inA.shape[1] - 1  # number of sequences
    # we will test using Carathéodory's theorem. Essentially, we need at most d+1 vertices to represent a vertex

    if r == 0:  # statement is vacuously true.
        return SolverResult({
            'eq_valid': True,
            'holder_ineqs_matrix': np.empty((0, 3, dim + 1)),
            'holder_ineqs_overall_powers': np.empty(0),
            'lp_mon_ineqs_matrix': np.empty((0, 2, dim + 1)),
            'lp_mon_ineqs_overall_powers': np.empty(0),
        })

    lin_constraints = np.empty([r, 0])
    is_mon_constr = []

    for k in range(r):
        relevant_points = inA.copy()[:, :-1]  # drop the final column (c-powers)
        # origin reflected about the point under consideration (subtract point k from all others)
        relevant_points += (np.eye(1, r, k=k).T - 1) * relevant_points[k]
        relevant_points = relevant_points.T  # transpose the matrix (for simplicity later)

        for j in range(dim + 1):  # consider all dimensions of simplexes

            # for each candidate simplex (to host a convex combination in the given dimension)
            for candidate_simplex in get_combs(r, j):
                m = null_space(relevant_points[:, candidate_simplex])

                # null space: must have dim 1 AND can't have all values 0 AND all values must have same sign
                if m.shape[1] == 1 and max(abs(m)) > 0 and (np.all(m > 0) if m[0] > 0 else np.all(m < 0)):

                    m /= sum(m)  # normalise for convex combination
                    if not np.sum(vector_element_is_zero_ish(m)):  # (not) any element is zero_ish

                        new_constraint = np.zeros(r)  # construct an r-length vector for the constraint

                        # convert the len(candidate_simplex)-vector to an r-length-vector
                        for i in range(len(candidate_simplex)):
                            new_val = np.squeeze(m[i])
                            if is_zero_ish(new_val):
                                new_val = 0
                            new_constraint[candidate_simplex[i]] = -new_val

                        is_mon_constr.append((not is_zero_ish(new_constraint[k]), k))

                        # "deals with both kinds of constraints on the ith point"
                        new_constraint[k] = 1 + 2 * new_constraint[k]
                        # append to existing constraints
                        lin_constraints = np.c_[lin_constraints, new_constraint]

    lin_constraints = lin_constraints.T
    b = np.zeros(lin_constraints.shape[0])
    c = inA[:, -1]

    res = linprog(-c, A_ub=-lin_constraints, b_ub=b, method='highs-ipm', bounds=(None, None))

    if (not res.success) or not np.all(vector_element_is_zero_ish(res.upper.marginals)): # !!!
        return SolverResult({
            'eq_valid': False,
            'holder_ineqs_matrix': None,
            'holder_ineqs_overall_powers': None,
            'lp_mon_ineqs_matrix': None,
            'lp_mon_ineqs_overall_powers': None,
        })

    ineq_resid = -res.ineqlin.marginals
    holder_c = []
    holder_l = []
    holder_p = []
    linear_c = []
    linear_l = []
    linear_p = []

    tight_constraints = [i for i in range(len(ineq_resid)) if not is_zero_ish(ineq_resid[i])]

    for i in tight_constraints:
        negative_points = np.flatnonzero(lin_constraints[i] < -ZERO_BOUND)
        ind = negative_points if not is_mon_constr[i][0] else np.setdiff1d(negative_points, is_mon_constr[i][1])
        c = np.squeeze(np.asarray(lin_constraints[i, ind]))
        p1 = np.tile(c, (dim, 1)).T
        p2 = inA[ind, :-1]

        p3 = np.cumsum(np.multiply(p1, p2), 0)
        p4 = 1 / np.cumsum(c)
        p = np.multiply(p3.T, p4).T

        for k in range(1, c.size):
            holder_c.append(np.array([np.squeeze(np.asarray(p[k - 1, :])), np.squeeze(np.asarray(inA[ind[k], :-1]))]))
            holder_l.append(c[k] / np.sum(c[:k + 1]))
            holder_p.append(-np.sum(c[0:k + 1]) * ineq_resid[i])

        if is_mon_constr[i][0]:
            linear_c.append(np.squeeze(np.asarray(inA[is_mon_constr[i][1], :-1])))
            linear_l.append(2 - 2 / (1 + lin_constraints[i, is_mon_constr[i][1]]))
            linear_p.append(ineq_resid[i] * lin_constraints[i, is_mon_constr[i][1]])

    holder_flat_matrix = np.empty((len(holder_l), 3 * (dim + 1) + 1))


    for j in range(len(holder_c)):
        for k in range(3):
            v = None
            indiv_power = None
            if k != 2:
                v = holder_c[j][k, :]
                indiv_power = holder_l[j]
            else:
                v = np.matmul(np.array([1 - holder_l[j], holder_l[j]]), holder_c[j])
                indiv_power = -1.0
            if k == 0:
                indiv_power = 1 - indiv_power

            holder_flat_matrix[j, k * (dim + 1):k * (dim + 1) + dim] = np.squeeze(np.asarray(v))
            holder_flat_matrix[j, k * (dim + 1) + dim] = indiv_power
        holder_flat_matrix[j, -1] = float(holder_p[j])

    lp_mon_flat_matrix = np.empty((len(linear_l), 2 * (dim + 1) + 1))

    for j in range(len(linear_c)):
        for k in range(2):
            v = None
            indiv_power = 1.0
            if k == 1:
                v = linear_c[j]
                indiv_power = -linear_l[j]
            else:
                v = linear_c[j] * linear_l[j]

            lp_mon_flat_matrix[j, k * (dim + 1):k * (dim + 1) + dim] = v
            lp_mon_flat_matrix[j, k * (dim + 1) + dim] = indiv_power
        if is_zero_ish(float(linear_l[j])):
            lp_mon_flat_matrix[j, 6] = 12893287919891 # !!! need to test edgecases
        else:
            lp_mon_flat_matrix[j, 6] = float(linear_p[j]) / float(linear_l[j])


    holder_flat_matrix = merge_common_matrix_expressions(holder_flat_matrix)
    lp_mon_flat_matrix = merge_common_matrix_expressions(lp_mon_flat_matrix)

    if round_near_zeroes:
        holder_flat_matrix = matrix_round_near_zeroes(holder_flat_matrix)
        lp_mon_flat_matrix = matrix_round_near_zeroes(lp_mon_flat_matrix)

    holder_ineqs_matrix = np.empty((len(holder_flat_matrix), 3, dim + 1))
    holder_ineqs_overall_powers = np.empty(len(holder_flat_matrix))
    for j in range(0, len(holder_flat_matrix)):
        a = holder_flat_matrix[j]
        holder_ineqs_matrix[j] = a[:-1].reshape((3, dim + 1))
        holder_ineqs_overall_powers[j] = a[-1]

    lp_mon_ineqs_matrix = np.empty((len(lp_mon_flat_matrix), 2, dim + 1))
    lp_mon_ineqs_overall_powers = np.empty(len(lp_mon_flat_matrix))
    for j in range(0, len(lp_mon_flat_matrix)):
        a = lp_mon_flat_matrix[j]
        lp_mon_ineqs_matrix[j] = a[:-1].reshape((2, dim + 1))
        lp_mon_ineqs_overall_powers[j] = a[-1]

    return SolverResult({
        'eq_valid': True,
        'holder_ineqs_matrix': holder_ineqs_matrix,
        'holder_ineqs_overall_powers': holder_ineqs_overall_powers,
        'lp_mon_ineqs_matrix': lp_mon_ineqs_matrix,
        'lp_mon_ineqs_overall_powers': lp_mon_ineqs_overall_powers,
    })


if __name__ == '__main__':
    inA = np.matrix([[1, 0, 0.5], [0, 1, 0.5], [0.5, 0.5, -1]], dtype=np.float64)
    solver_result = ineq_solver(inA)
    print(solver_result.eq_valid)
    print(solver_result.holder_ineqs_matrix)
    print(solver_result.holder_ineqs_overall_powers)
    print(solver_result.lp_mon_ineqs_matrix)
    print(solver_result.lp_mon_ineqs_overall_powers)