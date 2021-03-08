#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tefirman, austin clark, jonathan huihui
"""
#import sys
""" Conditions of Simulation and Inference """
timeInc = 300                  # Sampling time (delta T)
numDays = 7                    # Total time length of each simulation
numTrials = 100                # Number of trajectories in each simulation
maxN = 59                      # Max possible number of proteins (for FSP purposes) (old value = 48)
maxn = 10                      # Max possible number of mRNA (for FSP purposes)
numIterations = 200            # Number of frames to project forward for the likelihood (old value = 390)
simNum = 5                     # Index of which simulation to infer from
numFits = 1                    # Number of times to use the minimize function for parameter selection

""" Index of CUDA Enabled GPU Device """
gpuDevice = 0          

""" Filename of simulation to look for/create  """
fname = './Simulations/ToggleGillespie_RNA_7days_' + str(simNum) + '.npz'

""" Filename of parameter file to create/save """
fout = 'ExtractedParameters_ToggleGillespieParms_noRNA_GPU_timeInc300_VariableNumIterations' + str(simNum) + '.npz'

""" Filename of progress file """
fprogress = 'ProgressReport_ToggleGillespieParms_noRNA_GPU_timeInc300_VariableNumIterations' + str(simNum) + '.txt'

""" Python Modules Necessary for Function """
import cupy as cp  # Req for GPU
import datetime
from math import factorial
import numpy as np
import os
import scipy
from scipy.linalg.basic import solve  # Req for _solve_P_Q
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import matfuncs
from scipy.sparse.linalg._expm_multiply import _ident_like, _exact_1_norm as _onenorm
from scipy.special import comb
from scipy.optimize import minimize
""" Initialize context for CUPY CUDA device """
cp.cuda.Device(gpuDevice).use()
# ============================= BEGIN ARC CONTRIB =============================
# Code within this section adapted from files withing Scipy package
# scipy/sparse/linalg/matfuncs.py
# Pulled on 20200107
# commit hash:77e5b9b40a22fc6e565c9c5e14330d3ea8536ae8


def _GPU_matrix_product(A, B):
    """
    Helper function that copies data from CPU to GPU for calculating
    dot products on GPU.
    """
    # possible rounding error due to float32 single-precision
    # copy arrays to GPU device
    A_gpu0 = cp.array(A, dtype=cp.float32)
    B_gpu0 = cp.array(B, dtype=cp.float32)
    out_gpu0 = cp.dot(A_gpu0, B_gpu0)
    # move array from device to host
    return cp.asnumpy(out_gpu0)


class _ExpmPadeHelper(object):
    """
    Help lazily evaluate a matrix exponential.

    The idea is to not do more work than we need for high expm precision,
    so we lazily compute matrix powers and store or precompute
    other properties of the matrix.
    """
    # from matfuncs.py

    def __init__(self, A, structure=None, use_exact_onenorm=False):
        """
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        structure : str, optional
            A string describing the structure of matrix `A`.
            Only `upper_triangular` is currently supported.
        use_exact_onenorm : bool, optional
            If True then only the exact one-norm of matrix powers and products
            will be used. Otherwise, the one-norm of powers and products
            may initially be estimated.
        """
        self.A = A
        self._A2 = None
        self._A4 = None
        self._A6 = None
        self._A8 = None
        self._A10 = None
        self._d4_exact = None
        self._d6_exact = None
        self._d8_exact = None
        self._d10_exact = None
        self._d4_approx = None
        self._d6_approx = None
        self._d8_approx = None
        self._d10_approx = None
        self.ident = _ident_like(A)
        self.structure = structure
        self.use_exact_onenorm = use_exact_onenorm

    @property
    def A2(self):
        if self._A2 is None:
            self._A2 = _GPU_matrix_product(
                    self.A, self.A)
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            self._A4 = _GPU_matrix_product(
                    self.A2, self.A2)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            self._A6 = _GPU_matrix_product(
                    self.A4, self.A2)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            self._A8 = _GPU_matrix_product(
                    self.A6, self.A2)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            self._A10 = _GPU_matrix_product(
                    self.A4, self.A6)
        return self._A10

    @property
    def d4_tight(self):
        if self._d4_exact is None:
            self._d4_exact = _onenorm(self.A4)**(1/4.)
        return self._d4_exact

    @property
    def d6_tight(self):
        if self._d6_exact is None:
            self._d6_exact = _onenorm(self.A6)**(1/6.)
        return self._d6_exact

    @property
    def d8_tight(self):
        if self._d8_exact is None:
            self._d8_exact = _onenorm(self.A8)**(1/8.)
        return self._d8_exact

    @property
    def d10_tight(self):
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10)**(1/10.)
        return self._d10_exact

    @property
    def d4_loose(self):
        if self.use_exact_onenorm:
            return self.d4_tight
        if self._d4_exact is not None:
            return self._d4_exact
        else:
            if self._d4_approx is None:
                self._d4_approx = matfuncs._onenormest_matrix_power(self.A2, 2,
                        structure=self.structure)**(1/4.)
            return self._d4_approx

    @property
    def d6_loose(self):
        if self.use_exact_onenorm:
            return self.d6_tight
        if self._d6_exact is not None:
            return self._d6_exact
        else:
            if self._d6_approx is None:
                self._d6_approx = matfuncs._onenormest_matrix_power(self.A2, 3,
                        structure=self.structure)**(1/6.)
            return self._d6_approx

    @property
    def d8_loose(self):
        if self.use_exact_onenorm:
            return self.d8_tight
        if self._d8_exact is not None:
            return self._d8_exact
        else:
            if self._d8_approx is None:
                self._d8_approx = matfuncs._onenormest_matrix_power(self.A4, 2,
                        structure=self.structure)**(1/8.)
            return self._d8_approx

    @property
    def d10_loose(self):
        if self.use_exact_onenorm:
            return self.d10_tight
        if self._d10_exact is not None:
            return self._d10_exact
        else:
            if self._d10_approx is None:
                self._d10_approx = matfuncs._onenormest_product((self.A4, self.A6),
                        structure=self.structure)**(1/10.)
            return self._d10_approx

    def pade3(self):
        b = (120., 60., 12., 1.)
        U = _GPU_matrix_product(self.A,
                                b[3]*self.A2 + b[1]*self.ident)
        V = b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade5(self):
        b = (30240., 15120., 3360., 420., 30., 1.)
        U = _GPU_matrix_product(self.A,
                                b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident)
        V = b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade7(self):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = _GPU_matrix_product(self.A,
                                b[7]*self.A6 + b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident)
        V = b[6]*self.A6 + b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade9(self):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                2162160., 110880., 3960., 90., 1.)
        U = _GPU_matrix_product(self.A,
                (b[9]*self.A8 + b[7]*self.A6 + b[5]*self.A4 +
                    b[3]*self.A2 + b[1]*self.ident))
        V = (b[8]*self.A8 + b[6]*self.A6 + b[4]*self.A4 +
                b[2]*self.A2 + b[0]*self.ident)
        return U, V

    def pade13_scaled(self, s):
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        B = self.A * 2**-s
        B2 = self.A2 * 2**(-2*s)
        B4 = self.A4 * 2**(-4*s)
        B6 = self.A6 * 2**(-6*s)
        U2 = _GPU_matrix_product(B6,
                b[13]*B6 + b[11]*B4 + b[9]*B2)
        U = _GPU_matrix_product(B,
                (U2 + b[7]*B6 + b[5]*B4 +
                    b[3]*B2 + b[1]*self.ident))
        V2 = _GPU_matrix_product(B6,
                b[12]*B6 + b[10]*B4 + b[8]*B2)
        V = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident
        return U, V


def _ell(A, m):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    A : linear operator
        A linear operator whose norm of power we care about.
    m : int
        The power of the linear operator

    Returns
    -------
    value : int
        A value related to a bound.

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
    # They are coefficients of terms of a generating function series expansion.
    choose_2m_m = comb(2*m, m, exact=True)
    abs_c_recip = float(choose_2m_m * factorial(2*m + 1))

    # This is explained after Eq. (1.2) of the 2009 expm paper.
    # It is the "unit roundoff" of IEEE double precision arithmetic.
    u = 2**-53

    # Compute the one-norm of matrix power p of abs(A).
    A_abs_onenorm = matfuncs._onenorm_matrix_power_nnm(abs(A), 2*m + 1)

    # Treat zero norm as a special case.
    if not A_abs_onenorm:
        return 0

    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    log2_alpha_div_u = np.log2(alpha/u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)


def _solve_P_Q(U, V, structure=None):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade numerator.
    V : ndarray
        Pade denominator.
    structure : str, optional
        A string describing the structure of both matrices `U` and `V`.
        Only `upper_triangular` is currently supported.

    Notes
    -----
    The `structure` argument is inspired by similar args
    for theano and cvxopt functions.

    """
    P = U + V
    Q = -U + V
    P_gpu0 = cp.asarray(P)
    Q_gpu0 = cp.asarray(Q)
    # if isspmatrix(U) or is_pydata_spmatrix(U):
    #     return spsolve(Q, P)
    # elif structure is None:
    #     return solve(Q, P)
    # elif structure == UPPER_TRIANGULAR:
    #     return solve_triangular(Q, P)
    # else:
    #     raise ValueError('unsupported matrix structure: ' + str(structure))
    return cp.linalg.solve(Q_gpu0, P_gpu0)


def _sinch(x):
    """
    Stably evaluate sinch.

    Notes
    -----
    The strategy of falling back to a sixth order Taylor expansion
    was suggested by the Spallation Neutron Source docs
    which was found on the internet by google search.
    http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html
    The details of the cutoff point and the Horner-like evaluation
    was picked without reference to anything in particular.

    Note that sinch is not currently implemented in scipy.special,
    whereas the "engineer's" definition of sinc is implemented.
    The implementation of sinc involves a scaling factor of pi
    that distinguishes it from the "mathematician's" version of sinc.

    """

    # If x is small then use sixth order Taylor expansion.
    # How small is small? I am using the point where the relative error
    # of the approximation is less than 1e-14.
    # If x is large then directly evaluate sinh(x) / x.
    x2 = x*x
    if abs(x) < 0.0135:
        return 1 + (x2/6.)*(1 + (x2/20.)*(1 + (x2/42.)))
    else:
        return np.sinh(x) / x


def _eq_10_42(lam_1, lam_2, t_12):
    """
    Equation (10.42) of Functions of Matrices: Theory and Computation.

    Notes
    -----
    This is a helper function for _fragment_2_1 of expm_2009.
    Equation (10.42) is on page 251 in the section on Schur algorithms.
    In particular, section 10.4.3 explains the Schur-Parlett algorithm.
    expm([[lam_1, t_12], [0, lam_1])
    =
    [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],
    [0, exp(lam_2)]
    """

    # The plain formula t_12 * (exp(lam_2) - exp(lam_2)) / (lam_2 - lam_1)
    # apparently suffers from cancellation, according to Higham's textbook.
    # A nice implementation of sinch, defined as sinh(x)/x,
    # will apparently work around the cancellation.
    a = 0.5 * (lam_1 + lam_2)
    b = 0.5 * (lam_1 - lam_2)
    return t_12 * np.exp(a) * _sinch(b)


def _fragment_2_1(X, T, s):
    """
    A helper function for expm_2009.

    Notes
    -----
    The argument X is modified in-place, but this modification is not the same
    as the returned value of the function.
    This function also takes pains to do things in ways that are compatible
    with sparse matrices, for example by avoiding fancy indexing
    and by using methods of the matrices whenever possible instead of
    using functions of the numpy or scipy libraries themselves.

    """
    # Form X = r_m(2^-s T)
    # Replace diag(X) by exp(2^-s diag(T)).
    n = X.shape[0]
    diag_T = np.ravel(T.diagonal().copy())

    # Replace diag(X) by exp(2^-s diag(T)).
    scale = 2 ** -s
    exp_diag = np.exp(scale * diag_T)
    for k in range(n):
        X[k, k] = exp_diag[k]

    for i in range(s-1, -1, -1):
        X = X.dot(X)

        # Replace diag(X) by exp(2^-i diag(T)).
        scale = 2 ** -i
        exp_diag = np.exp(scale * diag_T)
        for k in range(n):
            X[k, k] = exp_diag[k]

        # Replace (first) superdiagonal of X by explicit formula
        # for superdiagonal of exp(2^-i T) from Eq (10.42) of
        # the author's 2008 textbook
        # Functions of Matrices: Theory and Computation.
        for k in range(n-1):
            lam_1 = scale * diag_T[k]
            lam_2 = scale * diag_T[k+1]
            t_12 = scale * T[k, k+1]
            value = _eq_10_42(lam_1, lam_2, t_12)
            X[k, k+1] = value

    # Return the updated X matrix.
    return X


def cudaExp(A):
    return _expm(A)


def expm(A):
    """
    Compute the matrix exponential using Pade approximation.
    Adapted from Scipy matfuncs.py v1.4.1
    """
    # return _expm(A, use_exact_onenorm='auto')
    return _expm(A)


def _expm(A):
    # def _expm(A, use_exact_onenorm):
    # Core of expm, separated to allow testing exact and approximate
    # algorithms.

    # Avoid indiscriminate asarray() to allow sparse or other strange arrays.
    if isinstance(A, (list, tuple, np.matrix)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    """
    Don't need this here. Not expecting size-0 input or trivial
    case of A.shape == (1,1)
    """
    # gracefully handle size-0 input,
    # carefully handling sparse scenario
    # if A.shape == (0, 0):
    #     out = np.zeros([0, 0], dtype=A.dtype)
    #     if isspmatrix(A) or is_pydata_spmatrix(A):
    #         return A.__class__(out)
    #     return out
    #
    # Trivial case
    # if A.shape == (1, 1):
    #     out = [[np.exp(A[0, 0])]]
    # #
    #     # Avoid indiscriminate casting to ndarray to
    #     # allow for sparse or other strange arrays
    #     if isspmatrix(A) or is_pydata_spmatrix(A):
    #         return A.__class__(out)
    #
    #     return np.array(out)

    # Ensure input is of float type, to avoid integer overflows etc.
    # if ((isinstance(A, np.ndarray) or isspmatrix(A) or is_pydata_spmatrix(A))
    #         and not np.issubdtype(A.dtype, np.inexact)):
    #     A = A.astype(float)

    # Detect upper triangularity.
    # structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None
    structure = None
    # if use_exact_onenorm == "auto":
    # Hardcode a matrix order threshold for exact vs. estimated one-norms.
    use_exact_onenorm = A.shape[0] < 200
    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper(
                A, structure=structure, use_exact_onenorm=use_exact_onenorm)
    # Try Pade order 3.
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
        U, V = h.pade3()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade order 5.
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
        U, V = h.pade5()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade orders 7 and 9.
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
        U, V = h.pade7()
        return _solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
        U, V = h.pade9()
        return _solve_P_Q(U, V, structure=structure)

    # Use Pade order 13.
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25

    # Choose smallest s>=0 such that 2**(-s) eta_5 <= theta_13
    if eta_5 == 0:
        # Nilpotent special case
        s = 0
    else:
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2**-s * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V, structure=structure)
    # structure is not upper triangular
    # if structure == UPPER_TRIANGULAR:
    #    # Invoke Code Fragment 2.1.
    #    X = _fragment_2_1(X, h.A, s)
    # else:
    # X = r_13(A)^(2^s) by repeated squaring.
    X_gpu0 = cp.array(X, dtype=cp.float32)
    for i in np.arange(s):
        # X = X.dot(X)
        X_gpu0 = cp.dot(X_gpu0, X_gpu0)
    return X


# ============================== END ARC CONTRIB ==============================


def conditionsInitGill(g,g_pro,g_rep,g_prorep,d,p,r,k_f,k_b,exclusive,\
n_A_init,n_a_init,n_alpha_init,n_B_init,n_b_init,n_beta_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all of your stated conditions """
    return {'g':g, 'g_pro':g_pro, 'g_rep':g_rep, 'g_prorep':g_prorep, \
            'd':d, 'p':p, 'r':r, 'k_f':k_f, 'k_b':k_b, 'exclusive':exclusive, \
            'N_A_init':n_A_init, 'N_a_init':n_a_init, 'N_alpha_init':n_alpha_init, \
            'N_B_init':n_B_init, 'N_b_init':n_b_init, 'N_beta_init':n_beta_init, \
            'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def gillespieSim(conditions,n_A,n_a,n_alpha,n_B,n_b,n_beta):
    """ Gillespie simulation used to generate the synthetic toggle switch input data """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_a = (conditions['N_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_alpha = (conditions['N_alpha_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_B = (conditions['N_B_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_b = (conditions['N_b_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_beta = (conditions['N_beta_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_a) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            numa = np.copy(conditions['N_a_init'])
            numalpha = np.copy(conditions['N_alpha_init'])
            numB = np.copy(conditions['N_B_init'])
            numb = np.copy(conditions['N_b_init'])
            numbeta = np.copy(conditions['N_beta_init'])
            n_A.append([np.copy(numA)])
            n_a.append([np.copy(numa)])
            n_alpha.append([np.copy(numalpha)])
            n_B.append([np.copy(numB)])
            n_b.append([np.copy(numb)])
            n_beta.append([np.copy(numbeta)])
        else:
            numA = np.copy(n_A[numTrial][-1])
            numa = np.copy(n_a[numTrial][-1])
            numalpha = np.copy(n_alpha[numTrial][-1])
            numB = np.copy(n_B[numTrial][-1])
            numb = np.copy(n_b[numTrial][-1])
            numbeta = np.copy(n_beta[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        tempData = open(fprogress,'a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            if conditions['exclusive']:
                prob = [conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numalpha*(1 - numbeta),\
                conditions['g_rep']*numbeta*(1 - numalpha),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numa,\
                conditions['p']*numa,\
                conditions['r']*numA,\
                conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numbeta*(1 - numalpha),\
                conditions['g_rep']*numalpha*(1 - numbeta),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numb,\
                conditions['p']*numb,\
                conditions['r']*numB,\
                conditions['k_f']*(1 - numalpha - numbeta)*numA,\
                conditions['k_b']*numalpha,\
                conditions['k_f']*(1 - numalpha - numbeta)*numB,\
                conditions['k_b']*numbeta]
            else:
                prob = [conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numalpha*(1 - numbeta),\
                conditions['g_rep']*numbeta*(1 - numalpha),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numa,\
                conditions['p']*numa,\
                conditions['r']*numA,\
                conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numbeta*(1 - numalpha),\
                conditions['g_rep']*numalpha*(1 - numbeta),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numb,\
                conditions['p']*numb,\
                conditions['r']*numB,\
                conditions['k_f']*(1 - numalpha)*numA,\
                conditions['k_b']*numalpha,\
                conditions['k_f']*(1 - numbeta)*numB,\
                conditions['k_b']*numbeta]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA).tolist())
                n_a[numTrial].append(np.copy(numa).tolist())
                n_alpha[numTrial].append(np.copy(numalpha).tolist())
                n_B[numTrial].append(np.copy(numB).tolist())
                n_b[numTrial].append(np.copy(numb).tolist())
                n_beta[numTrial].append(np.copy(numbeta).tolist())
                incCheckpoint += conditions['inc']
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 <= sum(prob[:4]):
                numa += 1
            elif randNum2 <= sum(prob[:5]):
                numa -= 1
            elif randNum2 <= sum(prob[:6]):
                numA += 1
            elif randNum2 <= sum(prob[:7]):
                numA -= 1
            elif randNum2 <= sum(prob[:11]):
                numb += 1
            elif randNum2 <= sum(prob[:12]):
                numb -= 1
            elif randNum2 <= sum(prob[:13]):
                numB += 1
            elif randNum2 <= sum(prob[:14]):
                numB -= 1
            elif randNum2 <= sum(prob[:15]):
                numalpha += 1
                numA -= 1
            elif randNum2 <= sum(prob[:16]):
                numalpha -= 1
                numA += 1
            elif randNum2 <= sum(prob[:17]):
                numbeta += 1
                numB -= 1
            else:
                numbeta -= 1
                numB += 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
        n_a[numTrial] = n_a[numTrial][:conditions['numSteps'] + 1]
        n_alpha[numTrial] = n_alpha[numTrial][:conditions['numSteps'] + 1]
        n_B[numTrial] = n_B[numTrial][:conditions['numSteps'] + 1]
        n_b[numTrial] = n_b[numTrial][:conditions['numSteps'] + 1]
        n_beta[numTrial] = n_beta[numTrial][:conditions['numSteps'] + 1]
    return n_A, n_a, n_alpha, n_B, n_b, n_beta

def peakVals(origHist,numFilter,minVal):
    """ Robustly identifies peak locations for 2D probability distributions """
    simHist = np.copy(origHist)
    for numTry in range(numFilter):
        for ind1 in range(len(simHist)):
            for ind2 in range(len(simHist[ind1])):
                simHist[ind1,ind2] = np.sum(simHist[max(ind1 - 1,0):min(ind1 + 2,len(simHist)),\
                max(ind2 - 1,0):min(ind2 + 2,len(simHist[ind1]))])/\
                np.size(simHist[max(ind1 - 1,0):min(ind1 + 2,len(simHist)),\
                max(ind2 - 1,0):min(ind2 + 2,len(simHist[ind1]))])
    maxInds = []
    for ind1 in range(len(simHist)):
        for ind2 in range(len(simHist[ind1])):
            if simHist[ind1,ind2] == np.max(simHist[max(ind1 - 1,0):min(ind1 + 2,len(simHist)),\
            max(ind2 - 1,0):min(ind2 + 2,len(simHist[ind1]))]) and simHist[ind1,ind2] >= minVal:
                maxInds.append([ind1,ind2])
    return maxInds

def entropyStats(n_A,n_B,maxInds):
    """ Calculates the different entropy values for a given set of trajectories """
    global maxN
    stateProbs = [np.zeros((maxN**2,maxN**2)) for ind in range(len(maxInds))]
    cgProbs = np.zeros((len(maxInds),len(maxInds)))
    dwellVals = [[] for ind in range(len(maxInds))]
    for numTrial in range(len(n_A)):
        tempData = open(fprogress,'a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        cgTraj = -1*np.ones(len(n_A[numTrial]))
        cgTraj[np.all([n_A[numTrial] <= maxInds[0][0],n_B[numTrial] >= maxInds[0][1]],axis=0)] = 0
        cgTraj[np.all([n_A[numTrial] >= maxInds[1][0],n_B[numTrial] <= maxInds[1][1]],axis=0)] = 1
        ind1 = np.where(cgTraj >= 0)[0][0]
        inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        while len(inds) > 0:
            stateProbs[int(cgTraj[ind1])] += np.histogram2d(maxN*n_B[numTrial][ind1 + 1:ind1 + inds[0]] + \
            n_A[numTrial][ind1 + 1:ind1 + inds[0]],maxN*n_B[numTrial][ind1:ind1 + inds[0] - 1] + \
            n_A[numTrial][ind1:ind1 + inds[0] - 1],bins=np.arange(-0.5,maxN**2))[0]
            cgProbs[int(cgTraj[ind1 + inds[0]]),int(cgTraj[ind1])] += 1
            cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += inds[0]
            dwellVals[int(cgTraj[ind1])].append(inds[0])
            ind1 += inds[0]
            inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        stateProbs[int(cgTraj[ind1])] += np.histogram2d(maxN*n_B[numTrial][ind1 + 1:] + \
        n_A[numTrial][ind1 + 1:],maxN*n_B[numTrial][ind1:-1] + n_A[numTrial][ind1:-1],\
        bins=np.arange(-0.5,maxN**2))[0]
        cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += len(n_A[numTrial]) - ind1 - 1
    totProbs = np.zeros((maxN**2,maxN**2))
    stateEntropies = []
    for ind in range(len(stateProbs)):
        totProbs += stateProbs[ind]
        stateProbs[ind] = stateProbs[ind]/np.sum(stateProbs[ind])
        stateEntropies.append(-np.nansum(stateProbs[ind]*np.log2(stateProbs[ind])))
    totProbs = totProbs/np.sum(totProbs)
    totEntropy = -np.nansum(totProbs*np.log2(totProbs))
    cgProbs = cgProbs/np.sum(cgProbs)
    macroEntropy = -np.nansum(cgProbs*np.log2(cgProbs))
    return totEntropy, stateEntropies, macroEntropy, dwellVals

def conditionsInitNoRNA(g,g_pro,g_rep,g_prorep,r,k_f,k_b,exclusive,\
n_A_init,n_alpha_init,n_B_init,n_beta_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all of your stated conditions """
    return {'g':g, 'g_pro':g_pro, 'g_rep':g_rep, 'g_prorep':g_prorep, \
            'r':r, 'k_f':k_f, 'k_b':k_b, 'exclusive':exclusive, \
            'N_A_init':n_A_init, 'N_alpha_init':n_alpha_init, \
            'N_B_init':n_B_init, 'N_beta_init':n_beta_init, \
            'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def gillespieSim_NoRNA(conditions,n_A,n_alpha,n_B,n_beta):
    """ Gillespie simulation used to generate trajectories using a reaction schema without mRNA """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_alpha = (conditions['N_alpha_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_B = (conditions['N_B_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_beta = (conditions['N_beta_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_alpha) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            numalpha = np.copy(conditions['N_alpha_init'])
            numB = np.copy(conditions['N_B_init'])
            numbeta = np.copy(conditions['N_beta_init'])
            n_A.append([np.copy(numA)])
            n_alpha.append([np.copy(numalpha)])
            n_B.append([np.copy(numB)])
            n_beta.append([np.copy(numbeta)])
        else:
            numA = np.copy(n_A[numTrial][-1])
            numalpha = np.copy(n_alpha[numTrial][-1])
            numB = np.copy(n_B[numTrial][-1])
            numbeta = np.copy(n_beta[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        tempData = open(fprogress,'a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            if conditions['exclusive']:
                prob = [conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numalpha*(1 - numbeta),\
                conditions['g_rep']*numbeta*(1 - numalpha),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['r']*numA,\
                conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numbeta*(1 - numalpha),\
                conditions['g_rep']*numalpha*(1 - numbeta),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['r']*numB,\
                conditions['k_f']*(1 - numalpha - numbeta)*numA,\
                conditions['k_b']*numalpha,\
                conditions['k_f']*(1 - numalpha - numbeta)*numB,\
                conditions['k_b']*numbeta]
            else:
                prob = [conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numalpha*(1 - numbeta),\
                conditions['g_rep']*numbeta*(1 - numalpha),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['r']*numA,\
                conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numbeta*(1 - numalpha),\
                conditions['g_rep']*numalpha*(1 - numbeta),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['r']*numB,\
                conditions['k_f']*(1 - numalpha)*numA,\
                conditions['k_b']*numalpha,\
                conditions['k_f']*(1 - numbeta)*numB,\
                conditions['k_b']*numbeta]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA).tolist())
                n_alpha[numTrial].append(np.copy(numalpha).tolist())
                n_B[numTrial].append(np.copy(numB).tolist())
                n_beta[numTrial].append(np.copy(numbeta).tolist())
                incCheckpoint += conditions['inc']
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 <= sum(prob[:4]):
                numA += 1
            elif randNum2 <= sum(prob[:5]):
                numA -= 1
            elif randNum2 <= sum(prob[:9]):
                numB += 1
            elif randNum2 <= sum(prob[:10]):
                numB -= 1
            elif randNum2 <= sum(prob[:11]):
                numalpha += 1
                numA -= 1
            elif randNum2 <= sum(prob[:12]):
                numalpha -= 1
                numA += 1
            elif randNum2 <= sum(prob[:13]):
                numbeta += 1
                numB -= 1
            else:
                numbeta -= 1
                numB += 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
        n_alpha[numTrial] = n_alpha[numTrial][:conditions['numSteps'] + 1]
        n_B[numTrial] = n_B[numTrial][:conditions['numSteps'] + 1]
        n_beta[numTrial] = n_beta[numTrial][:conditions['numSteps'] + 1]
    return n_A, n_alpha, n_B, n_beta


def noRNA_FSP_mle(reactionRates):
    """ Uses Finite State Projection to calculate the likelihood of """
    """ an input trajectory occurring in a reaction schema without mRNA. """
    """ reactionRates = ['g_pro','g_rep','r','k_f','k_b'] """
    global probs
    global maxN
    global numIterations
    global timeInc
    global timeVals
    if np.any(np.array(reactionRates) < 0):
        return float('Inf')
    probMatrix = np.zeros((4*(maxN**2),4*(maxN**2)))
    for probN_A in range(maxN):
        for probN_B in range(maxN):
            for probN_alpha in range(2):
                for probN_beta in range(2):
                    probMatrix[probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B]\
                    [probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B] -= \
                    (reactionRates[0]*(1 - probN_alpha)*(1 - probN_beta) + \
                    reactionRates[0]*probN_alpha*(1 - probN_beta) + \
                    reactionRates[1]*(1 - probN_alpha)*probN_beta + \
                    reactionRates[1]*probN_alpha*probN_beta + \
                    reactionRates[2]*probN_A + \
                    reactionRates[0]*(1 - probN_alpha)*(1 - probN_beta) + \
                    reactionRates[0]*(1 - probN_alpha)*probN_beta + \
                    reactionRates[1]*probN_alpha*(1 - probN_beta) + \
                    reactionRates[1]*probN_alpha*probN_beta + \
                    reactionRates[2]*probN_B + \
                    reactionRates[3]*(1 - probN_alpha - probN_beta)*probN_A + \
                    reactionRates[4]*probN_alpha + \
                    reactionRates[3]*(1 - probN_alpha - probN_beta)*probN_B + \
                    reactionRates[4]*probN_beta)
                    if probN_A <= maxN - 2:
                        probMatrix[probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + (probN_A + 1)*maxN + probN_B]\
                        [probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[0]*(1 - probN_alpha)*(1 - probN_beta) + \
                        reactionRates[0]*probN_alpha*(1 - probN_beta) + \
                        reactionRates[1]*(1 - probN_alpha)*probN_beta + \
                        reactionRates[1]*probN_alpha*probN_beta
                    if probN_A > 0:
                        probMatrix[probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + (probN_A - 1)*maxN + probN_B]\
                        [probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[2]*probN_A
                    if probN_B <= maxN - 2:
                        probMatrix[probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B + 1]\
                        [probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[0]*(1 - probN_alpha)*(1 - probN_beta) + \
                        reactionRates[0]*(1 - probN_alpha)*probN_beta + \
                        reactionRates[1]*probN_alpha*(1 - probN_beta) + \
                        reactionRates[1]*probN_alpha*probN_beta
                    if probN_B > 0:
                        probMatrix[probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B - 1]\
                        [probN_alpha*2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[2]*probN_B
                    if probN_alpha == 0 and probN_A > 0:
                        probMatrix[2*(maxN**2) + probN_beta*(maxN**2) + (probN_A - 1)*maxN + probN_B]\
                        [probN_beta*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[3]*(1 - probN_alpha - probN_beta)*probN_A
                    if probN_alpha == 1 and probN_A <= maxN - 2:
                        probMatrix[probN_beta*(maxN**2) + (probN_A + 1)*maxN + probN_B]\
                        [2*(maxN**2) + probN_beta*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[4]*probN_alpha
                    if probN_beta == 0 and probN_B > 0:
                        probMatrix[probN_alpha*2*(maxN**2) + maxN**2 + probN_A*maxN + probN_B - 1]\
                        [probN_alpha*2*(maxN**2) + probN_A*maxN + probN_B] += \
                        reactionRates[3]*(1 - probN_alpha - probN_beta)*probN_B
                    if probN_beta == 1 and probN_B <= maxN - 2:
                        probMatrix[probN_alpha*2*(maxN**2) + probN_A*maxN + probN_B + 1]\
                        [probN_alpha*2*(maxN**2) + maxN**2 + probN_A*maxN + probN_B] += \
                        reactionRates[4]*probN_beta
    probMatrix = np.hstack((probMatrix,np.zeros((4*(maxN**2),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    startTime = datetime.datetime.now()
    """ Run GPU matrix exponential """
    result_gpu = cudaExp(probMatrix*timeInc*numIterations)
    endTime = datetime.datetime.now()
    timeVals.append((endTime - startTime).total_seconds())
    numSteps = 100
    binaryString = np.binary_repr(numSteps)
    Z,q,t = result_gpu,0,len(binaryString)
    while binaryString[t - q - 1] == '0':
        Z = cp.dot(Z, Z)
        q += 1
    equil_gpu = Z
    for k in range(q + 1,t):
        Z = cp.dot(Z, Z)
        if binaryString[t-k-1] == '1':
            equil_gpu = cp.dot(equil_gpu, Z)
    startProb = np.vstack((np.vstack(((np.sum(probs[:-1,:-1],axis=1)/np.sum(probs[:-1,:-1])).reshape((maxN**2,1)),\
    (np.sum(probs[:-1,:-1],axis=1)/np.sum(probs[:-1,:-1])).reshape((maxN**2,1)))),\
    np.vstack(((np.sum(probs[:-1,:-1],axis=1)/np.sum(probs[:-1,:-1])).reshape((maxN**2,1)),\
    (np.sum(probs,axis=1)/np.sum(probs)).reshape((maxN**2 + 1,1))))))
    startProb = startProb/np.sum(startProb)
    startProb = np.array(startProb.astype(np.float32))
    # holder = cudamat.CUDAMatrix(startProb)
    holder = cp.asarray(startProb)
    # equilProb = cudamat.dot(equil_gpu,holder).asarray()[:-1].reshape(2,2,maxN,maxN)
    equilProb = cp.asnumpy(cp.dot(equil_gpu, holder))[:-1].reshape(2,2,maxN,maxN)
    for startN_A in range(maxN):
        for startN_B in range(maxN):
            equilProb[:,:,startN_A,startN_B] = equilProb[:,:,startN_A,startN_B]/np.sum(equilProb[:,:,startN_A,startN_B])
    equilProb = equilProb.reshape(4*(maxN**2))
    # result_gpu = result_gpu.asarray()
    result_gpu = cp.asnumpy(result_gpu)
    for ind in range(result_gpu.shape[0]):
        result_gpu[ind,:-1] *= equilProb
    transitionProb = np.zeros((maxN**2 + 1,maxN**2 + 1))
    for ind1 in range(4):
        for ind2 in range(4):
            transitionProb[:maxN**2,:maxN**2] += result_gpu[ind1*(maxN**2):(ind1 + 1)*(maxN**2),ind2*(maxN**2):(ind2 + 1)*(maxN**2)]
    loglike = -1*np.nansum(np.nansum(np.log(transitionProb)*probs.toarray()))
    if loglike != -0.0:
    	tempData = open(fprogress,'a')
    	tempData.write('g = g_pro = ' + str(round(reactionRates[0],3)) + ', g_rep = g_prorep = ' + str(round(reactionRates[1],3)) + \
    ', r = ' + str(round(reactionRates[2],4)) + ', k_f = ' + str(round(reactionRates[3],4)) + \
    ', k_b = ' + str(round(reactionRates[4],4)) + ', loglike = ' + str(round(loglike,1)) + '\n')
    	tempData.close()
    elif loglike == -0.0:
        loglike = np.inf
    np.savez_compressed(fout,timeVals=timeVals)


    return loglike

""" Defining Reaction Rates """

#conditions_Gill = conditionsInitGill(g,g_pro,g_rep,g_prorep,d,p,r,k_f,k_b,exclusive,\
#n_A_init,n_a_init,n_alpha_init,n_B_init,n_b_init,n_beta_init,inc,numSteps,numTrials)
conditions_Gill = conditionsInitGill(0.5,0.5,0.0025,0.0025,0.5,0.02,0.001,\
3.5e-06,2.0e-05,True,5,5,0,5,5,0,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" If simulations are already present, load them """
""" If not, create them using the Gillespie function above """

# if os.path.exists('GillespieSims_' + simConditions + '/GillespieSim' + str(simNum) + '.npz'):
tempVars = np.load(fname)
n_A_Gill = tempVars['n_A_Gill']
n_B_Gill = tempVars['n_B_Gill']
n_A_Gill = n_A_Gill[:,range(0,24*3600*numDays,timeInc)]
n_B_Gill = n_B_Gill[:,range(0,24*3600*numDays,timeInc)]
del tempVars

""" Running Metrics on Gillespie Input Trajectories """

simHist_Gill = np.zeros((maxN,maxN))
for numTrial in range(len(n_A_Gill)):
    simHist_Gill += np.histogram2d(n_A_Gill[numTrial],n_B_Gill[numTrial],bins=np.arange(-0.5,maxN))[0]
del numTrial
simHist_Gill = simHist_Gill/np.sum(simHist_Gill)
maxInds_Gill = peakVals(simHist_Gill[:75,:75],5,0.0001)
if maxInds_Gill[1][1] != 0:
    maxInds_Gill.append(maxInds_Gill.pop(1))
totEntropy_Gill,stateEntropies_Gill,macroEntropy_Gill,dwellVals_Gill = entropyStats(n_A_Gill,n_B_Gill,maxInds_Gill)
avgDwells_Gill = []
avgTotDwell_Gill = []
for ind in range(len(dwellVals_Gill)):
    avgDwells_Gill.append(np.average(dwellVals_Gill[ind])*conditions_Gill['inc'])
    avgTotDwell_Gill.extend(dwellVals_Gill[ind])
del ind
avgTotDwell_Gill = np.average(avgTotDwell_Gill)*conditions_Gill['inc']
if type(numIterations) == str:
    numIterations = int(round(avgTotDwell_Gill/conditions_Gill['inc']))

""" Calculating the Transition Frequencies for Maximum Likelihood Calculations """

probs = scipy.sparse.csc_matrix((np.ones(np.size(n_A_Gill[:,numIterations::numIterations])),\
(maxN*n_A_Gill[:,numIterations::numIterations].reshape(np.size(n_A_Gill[:,numIterations::numIterations])) + \
n_B_Gill[:,numIterations::numIterations].reshape(np.size(n_B_Gill[:,numIterations::numIterations])),\
maxN*n_A_Gill[:,:-numIterations:numIterations].reshape(np.size(n_A_Gill[:,:-numIterations:numIterations])) + \
n_B_Gill[:,:-numIterations:numIterations].reshape(np.size(n_B_Gill[:,:-numIterations:numIterations])))),shape=(maxN**2 + 1,maxN**2 + 1))


""" Fitting Optimal Parameters Using Maximum Likelihood Procedure """

startTime_Tot = datetime.datetime.now()
timeVals = []

startGuess = []
finalGuess = []
loglike = []
for numFit in range(numFits):
    """ Using different starting seeds to ensure we are finding the truly optimized solution... """
    tempData = open(fprogress,'a')
    tempData.write('Fit #' + str(numFit + 1) + '\n')
    tempData.close()
    startGuess.append(np.array([0.02,1e-4,0.001,3e-6,1.5e-5]))
    res = minimize(noRNA_FSP_mle,startGuess[numFit],method='nelder-mead',tol=0.1,options={'disp':True,'maxiter':500})
    finalGuess.append(res['x'])
    loglike.append(res['fun'])
    """ Saving progress as we go... """
    np.savez_compressed(fout,\
    conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
    maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
    macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
    dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
    avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
    finalGuess=finalGuess, timeVals=timeVals)
del numFit

""" Keeping the one with the lowest negative log likelihood, and therefore highest likelihood... """
bestGuess = finalGuess[np.where(loglike == np.min(loglike))[0][0]]
conditions_noRNA = conditionsInitNoRNA(bestGuess[0],bestGuess[0],bestGuess[1],bestGuess[1],\
bestGuess[2],bestGuess[3],bestGuess[4],True,18,1,0,0,timeInc,int((24*3600*numDays)/timeInc),numTrials)

endTime_Tot = datetime.datetime.now()
fitTime = endTime_Tot - startTime_Tot
del startTime_Tot, endTime_Tot

np.savez_compressed(fout,\
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, bestGuess=bestGuess, \
conditions_noRNA=conditions_noRNA, \
timeVals=timeVals, fitTime=fitTime)

