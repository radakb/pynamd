from __future__ import division, print_function

from numpy import asarray, zeros, identity, log, exp, sqrt, abs, min,\
        diagflat, diagonal, hstack, vstack, int32, float64, any, all, where,\
        nonzero, newaxis, ones, tile
from numpy.linalg import solve, inv
from scipy.optimize import minimize
from scipy.misc import logsumexp


class MSMLE(object):
    """Multi-State Maximum Likelihood Estimation (MSMLE)

    References:
    -----------
    1) Z. Tan, E. Gallicchio, M. Lapelosa, and R. M. Levy, "Theory of binless
      multi-state free energy estimation with applications to protein-ligand
      binding," J. Chem. Phys. 136, 144102 (2012).
    2) M. R. Shirts and J. D. Chodera, "Statistically optimal analysis of 
      samples from multiple equilibrium states," J. Chem. Phys. 129, 124105 
      (2008).
    """
    def __init__(self, reduced_potentials, nsamples):
        #   Store the sample size information. The PIs matrix is defined as
        # done by Tan, et al. (see Ref. 1, Appendix A).
        #
        self.nsamples = asarray(nsamples, int32)
        if self.nsamples.ndim != 1:
            raise ValueError('\'nsamples\' must be a one dimensional'
                             ' array-like object of integers.')
        if any(self.nsamples < 0):
            raise ValueError('\'nsamples\' must all be either zero or'
                             ' positive')
        self.total_samples = self.nsamples.sum()
        self.PIs = diagflat(self.nsamples_nonzero / self.total_samples)
        #   Arrange the data in the "flattened" sample space suggested in 
        # Ref 1. This is essentially a matrix where each element on the first 
        # axis is a data point with the potential evaluated in each state. No 
        # reference is made to where each point was sampled from. The advantage
        # of this is that the resulting array is 2D, not 3D (as in pymbar), and
        # requires no masking procedures.
        #   The energies are sorted into two arrays - one for the sampled 
        # states (_u_nj_m) and one for the unsampled states (_u_nj_k). This
        # (admittedly opaque) notation derives from the notation in Ref 1
        # whereby m sampled states and k unsampled states are considered. In 
        # any even this separation of data greatly simplifies calculation of 
        # the objective functions.
        #
        #TODO: Make 2D v 3D input data be an option?
        _u_ijn = asarray(reduced_potentials, float64)
        if _u_ijn.shape != (self.nstates_sampled, self.nstates, 
                            self.nsamples.max()):
            M, L, N = self.nstates_sampled, self.nstates, self.nsamples.max()
            raise ValueError("'reduced_energies' expected as %d x %d x %d "
                             'array-like object of floats'%(M, L, N)) 
        # Sort the energies from sampled and unsampled states.
        self._u_nj_m = zeros((self.total_samples, self.nstates_sampled))
        self._u_nj_k = zeros((self.total_samples, self.nstates_unsampled))
        for i, n in enumerate(self.nsamples_nonzero):
            j = self.nsamples_nonzero[:i].sum()
            self._u_nj_m[j:(j+n), :] += _u_ijn[i, self.mask_nonzero, :n].T
            self._u_nj_k[j:(j+n), :] += _u_ijn[i, self.mask_zero, :n].T
        # Since we only use these for energy _differences_, an arbitrary
        # constant shift is permitted and also generally desireable for 
        # avoiding under/overflow. In order to be totally sure that unsampled 
        # states do not affect the answer numerically, the shift _cannot_ 
        # depend on that data.
        #
        shift = self._u_nj_m.min()
        self._u_nj_m -= shift
        self._u_nj_k -= shift
        del _u_ijn

    @property
    def nstates(self):
        """The number of states for which weights will be computed."""
        return self.nsamples.size

    @property
    def mask_nonzero(self):
        """1d-array of indices for sampled states."""
        return nonzero(self.nsamples)[0]

    @property
    def nsamples_nonzero(self):
        """1d-array of samples greater than zero."""
        return self.nsamples[self.mask_nonzero]

    @property
    def nstates_sampled(self):
        """The number of states that have been sampled."""
        return self.mask_nonzero.size

    @property
    def mask_zero(self):
        """1d-array of indices for unsampled states."""
        return where(self.nsamples == 0)[0]

    @property
    def nstates_unsampled(self):
        """The number of states that have not been sampled."""
        return self.mask_zero.size

    @property
    def PIsdiag(self):
        return diagonal(self.PIs)

    def _objective_f(self, fguess=None):
        """Get a free energy array for the objective function."""
        if fguess is None:
            return zeros(self.nstates_sampled - 1)
        else:
            if fguess.size == self.nstates:
                return (fguess[self.mask_nonzero][1:] 
                        - fguess[self.mask_nonzero][0])
            elif fguess.size == self.nstates_sampled:
                return fguess[1:] - fguess[0]
            else:
                raise ValueError('Bad initial fguess shape')

    @property
    def _u_nj(self):
        """The sample sorted reduced potential at all states.

        This uses the internal sorting conventions and may not look at all like
        the input potential.
        """
        return hstack((self._u_nj_m, self._u_nj_k))

    @property
    def u_nj(self):
        """The reduced potential at all states."""
        u_nj = zeros((self.total_samples, self.nstates))
        u_nj[:, self.mask_nonzero] = self._u_nj_m
        u_nj[:, self.mask_zero] = self._u_nj_k
        return u_nj

    @property
    def _f_opt(self):
        """The sample sorted reduced free energies in all states."""
        mask0, maskn0 = self.mask_zero, self.mask_nonzero
        _f_opt = hstack((self.f_opt[maskn0], self.f_opt[mask0]))
        _f_opt -= _f_opt[0]
        return _f_opt

    @property
    def W_nj(self):
        """Compute the normalized sample weights.
        
        NB: This is the definition of the W matrix by Tan, et al. (see Ref. 1,
        Appendices A and B) _not_ the definition used by Shirts and Chodera
        (see Ref. 2, Eqn. 13). The only difference is a constant factor.

        Consider N samples drawn from M states. Let A_n be an array of all N
        samples such that the sample weights in each state have been computed
        (it no longer matters which state the sample came from). The average of
        A in each state, A_j, can be computed as:
      
        (W_nj*A_n[:, newaxis]).mean(axis=0)

        which is the same as

        (W_nj*A_n[:, newaxis]).sum(axis=0) / N

        Hence Shirts and Chodera fold the 1 / N factor into W_nj in their
        definition.
        """
        m = self.nstates_sampled
        _W_nj = self._W_nj
        W_nj = zeros(_W_nj.shape)
        W_nj[:, self.mask_nonzero] = _W_nj[:, :m]
        W_nj[:, self.mask_zero] = _W_nj[:, m:]
        return W_nj

    @property
    def _W_nj(self):
        """The sample sorted sample weights in all states."""
        m = self.nstates_sampled
        logQ_nj = self._f_opt[newaxis, :] - self._u_nj
        logNorm_n = logsumexp(logQ_nj[:, :m], 1, self.PIsdiag[newaxis, :])
        _W_nj = exp(logQ_nj - logNorm_n[:, newaxis])     
        return _W_nj

    @property
    def _R_nj(self):
        R_nj = zeros((self.total_samples, self.nstates_sampled))
        ncum = hstack((zeros(1, int32), self.nsamples_nonzero.cumsum()))
        for j, (n1, n2) in enumerate(zip(ncum[:-1], ncum[1:])):
            R_nj[n1:n2, j] += self.total_samples / (n2 - n1) 
        return R_nj

    def compute_expectations(self, A_jn, doerror=True):
        if A_jn.ndim == 2:
            # Assume same layout as reduced potential input
            assert A_jn.shape[0] == self.nstates_sampled
            if all(self.nsamples == self.nsamples[0]):
                # Shortcut for all elements containing data.
                A_n = A_jn.ravel()
            else:
                A_n = zeros(self.total_samples)
                for i, n in enumerate(self.nsamples_nonzero):
                    j = self.nsamples_nonzero[:i].sum()
                    A_n[j:(j+n)] = A_jn[i, :n]
        else:
            # As a flat array (since the originating state doesn't matter).
            A_n = A_jn

        A_j = (self.W_nj*A_n[:, newaxis]).mean(axis=0)
        varA_j = zeros(self.nstates)
        if doerror:
            # There are a number of errors in Ref 1. First, the definitions of
            # W and WA are incorrect, the extra factors of exp(f) should indeed
            # be included. Doing so obviates the need for the C matrix defined
            # therein. This itself is used incorrectly in that paper since the
            # dimensions are inconsistent during matrix multiplication.
            #
            # NB This is all borne out by R code released with Ref 1, which
            # uses the same equations below, but with completely different
            # notation (A1 --> G, B1 --> H). The matrix D in Ref 1 is NOT the
            # same as in the code, where it seems to be the first term in the
            # B matrix from the paper.
            #
            # Shorthand indices - notation similiar to Ref 1
            n, m, mpk = self.total_samples, self.nstates_sampled, self.nstates 
            mpk2 = 2*mpk
            mask0, maskn0 = self.mask_zero, self.mask_nonzero
            # Shuffle indices and re-define W (WWA, hereafter). 
            _W_nj = self._W_nj
            _WA_nj = _W_nj*A_n[:, newaxis]
            _A_j = _WA_nj.mean(axis=0)
            WWA_nj = hstack((_W_nj, _WA_nj))
            # Repeat the same procedure for free energies with the new W. 
            O = WWA_nj.T.dot(WWA_nj) / n
            Os = O[:, :m]
            D = hstack((Os.dot(self.PIs), zeros((mpk2, mpk2-m))))
            B1 = (D - identity(mpk2))[1:, 1:]
            A1 = (O - D[:, :m].dot(Os.T))[1:, 1:]
            V = solve(B1, A1).dot(inv(B1.T)) / n
            # This is how things are computed in the R code. The solution of
            # solve(B1, A1) is not numerically the same as B1invA1, although
            # they are supposedly mathematically identical. Non-exhaustive
            # tests show that these give virtually identical results for V,
            # except for state 0.
            #
#            B1invA1 = (-O + tile(O[0], O[0].size).reshape(O.shape))[1:, 1:]
#            V = B1invA1.dot(inv(B1.T)) / n
            U = zeros((mpk2, mpk2))
            U[1:, 1:] = V
            Ch = hstack((diagflat(-_A_j), identity(mpk)))
            V_full = Ch.dot(U).dot(Ch.T)
            varA_j[maskn0] = diagonal(V_full)[:m]
            varA_j[mask0] = diagonal(V_full)[m:]
        return A_j, varA_j

    @property
    def fvar(self):
        """Compute the covariance matrix of reduced free energies.

        Tan, et al. propose two formulas for the covariance matrix (V). The 
        first (Ref. 1, Eqn. B1) appears to be more numerically stable but is
        slightly more involved. The second is apparently faster but might
        yield nonsensical results. Here we default to the latter and fall
        back to the former if things look strange (i.e. negative variances).
        """
        # Shorthand indices - notation similiar to Ref. 1
        n, m, mpk = self.total_samples, self.nstates_sampled, self.nstates
        k = mpk - m
        mask0, maskn0 = self.mask_zero, self.mask_nonzero
        _W_nj = self._W_nj
        O = _W_nj.T.dot(_W_nj) / n 
        Os = O[:, :m]
        B1 = (hstack((Os.dot(self.PIs), zeros((mpk, k)))) 
              - identity(mpk))[1:, 1:]
        A1 = (O - Os.dot(self.PIs).dot(Os.T))[1:, 1:]
        V = solve(B1, A1).dot(inv(B1.T)) / n
#        if any(diagonal(V) < 0.):
#            # Not entirely sure about this...
#            D = _W_nj.T.dot(self._R_nj) / n
#            A1 = (O - D.dot(self.PIs).dot(D.T))[1:, 1:]
#            V = solve(B1, A1).dot(inv(B1.T)) / n
        # Unshuffle the state indices. Note that the variance of state 0 is
        # zero by construction and thus is omitted from V - since the user 
        # selected state 0 may be different from the lowest indexed sample 
        # state, re-adjust so that the actual state 0 has zero variance.
        V_full = zeros((mpk, mpk))
        V_full[1:, 1:] = V
        var = zeros(mpk)
        var[maskn0] = diagonal(V_full)[:m]
        var[mask0] = diagonal(V_full)[m:]
        var += var[0]
        var[0] = 0
        return var

    def _uwham_obj_grad(self, f_i):
        """Return the log-likelihood (scalar objective function) and its 
        gradient (wrt the free energies) for a given value of the free 
        energies.  Note that this expects one less free energy than there are
        states, since we always solve under the constraint that f_1 = 0.
        """
        _f_i = hstack((zeros(1), asarray(f_i)))
        # For numerical stability, use log quantities.
        logQ_nj = _f_i - self._u_nj_m
        logNorm_n = logsumexp(logQ_nj, 1, self.PIsdiag[newaxis, :])
        W_nj = exp(logQ_nj - logNorm_n[:, newaxis])
        # Compute matrix products and sums (equivalent to multiplying by
        # appropriate vector of ones). Note that using dot() with ndarrays is
        # _much_ faster than multiplying matrix objects.
        PIW = self.PIs.dot(W_nj.T)
        WPI = W_nj.dot(self.PIs)
        g = PIW.mean(axis=1) # used in gradient and Hessian computation
        kappa = logNorm_n.mean() - (self.PIsdiag.dot(_f_i)).sum()
        grad = (g - self.PIsdiag)[1:]
        self._hess = (diagflat(g) - PIW.dot(WPI) / self.total_samples)[1:, 1:]
        return kappa, grad

    def _uwham_hess(self, f_i=None):
        """Dummy function for minimizers that require a separate Hessian call.
        """
        return self._hess

    def solve_uwham(self, f_guess=None, method='trust-ncg', tol=1e-9, 
                    verbose=False):
        """Solve for the free energies by minimizing the scalar log-likelihood 
        function.  This corresponds to the "UWHAM" method of Tan, et al.

        f_guess : array-like
            Initial guess of the free energies.  If None, use all zeroes.
        method : str
            Minimization method.  This is passed directly to 
            scipy.optimize.minimize.  See the scipy documentation for details.
        tol : float, optional
            Tolerance for termination (see scipy.optimize.minimize).
        """
        _f_guess = self._objective_f(f_guess)
        jac_methods = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP',
                       'dogleg', 'trust-ncg']
        hess_methods = ['Newton-CG', 'dogleg', 'trust-ncg']
        _jac = (method in jac_methods)
        _hess = (self._uwham_hess if method in hess_methods else None)
        soltn = minimize(self._uwham_obj_grad, _f_guess, method=method, 
                         jac=_jac, hess=_hess, tol=float(tol), 
                         options={'disp': verbose})
        #   Now estimate the unsampled state free energies and map everything 
        # back to the original indices. Note that the lowest index of the 
        # sampled states was constrained to zero.
        #
        _f = hstack((zeros(1), soltn.x))
        self.f_opt = zeros(self.nstates)
        self.f_opt[self.mask_nonzero] = _f 
        f_unsampled = self._compute_unsampled_free_energies(self._u_nj_k)
        self.f_opt[self.mask_zero] = f_unsampled
        self.f_opt -= self.f_opt[0]
        self.check_sample_weight_normalization()
        return soltn 

    def _compute_unsampled_free_energies(self, reduced_potential):
        m, n = self.nstates_sampled, self.nsamples_nonzero
        logq_nj = self._f_opt[:m] - self._u_nj_m
        logw_n = -(reduced_potential + logsumexp(logq_nj, 1, n)[:, newaxis])
        f_unsampled = -logsumexp(logw_n, axis=0)
        return f_unsampled

    def check_sample_weight_normalization(self, tol=1e-6):
        m, W_nj = self.nstates_sampled, self._W_nj
        norm0 = W_nj.mean(axis=0) 
        if any(abs(norm0 - 1) > float(tol)):
            print('WARNING: Bad weight normalization on axis 0', norm0)
            return False
        norm1 = (W_nj[:, :m]*self.PIsdiag[newaxis, :]).sum(axis=1)
        if any(abs(norm1 - 1) > float(tol)):
            print('WARNING: Bad weight normalization on axis 1', norm1)
            return False
        return True

    def check_derivs(self, f_i=None, eps=1e-6):
        """Check analytic vs numerical derivatives at the given free energies.
        The numerical gradient and Hessian are computed by central finite
        differencing of the analytic objective and gradient, respectively.

        f_i : array-like (optional)
            Free energies to test at.  If None, use all zeroes.
        eps : float
            Displacement value for finite differencing.
        """
        _f_i = self._objective_f(f_i)
        grad_numer, hess_numer = [], []
        for i in xrange(self.nstates_sampled - 1):
            df = _f_i 
            df[i] += 0.5*eps
            kp, gp = self._uwham_obj_grad(df)
            df[i] -= eps
            km, gm = self._uwham_obj_grad(df)
            grad_numer.append((kp - km) / eps)
            hess_numer.append((gp - gm) / eps)
        grad_numer, hess_numer = asarray(grad_numer), asarray(hess_numer)
        k, grad_anal = self._uwham_obj_grad(_f_i)
        hess_anal = self.hess()
        return (grad_anal, grad_numer, hess_anal, hess_numer)

