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
        """
          Arrange the data in the "flattened" sample space suggested in Ref 1.
        This is essentially a matrix where each element on the first axis is a
        data point with the potential evaluated in each state. No reference is
        made to where each point was sampled from. The advantage of this is
        that the resulting array is 2D, not 3D (as in pymbar), and requires no
        masking procedures.
          The energies are sorted into two arrays - one for the sampled states
        (_u_nj_m) and one for the unsampled states (_u_nj_k). This (admittedly
        opaque) notation derives from the notation in Ref 1 whereby m sampled
        states and k unsampled states are considered. In any event this
        separation of data greatly simplifies calculation of the objective
        functions. 
        """
        _u_ijn = asarray(reduced_potentials, float64)
        M, L, N = self.nstates_sampled, self.nstates, self.nsamples.max()
        msg2 = ("'reduced_energies' expected as %d x %d array-like object"
                 'of floats'%(L, self.total_samples))
        msg3 = ("'reduced_energies' expected as %d x %d x %d array-like object"
                 'of floats'%(M, L, N))
        if _u_ijn.ndim == 2 and _u_ijn.shape != (L, self.total_samples):
            raise ValueError(msg2)
        elif _u_ijn.ndim == 3 and _u_ijn.shape != (M, L, N):
            raise ValueError(msg3)

        # Sort the energies from sampled and unsampled states.
        self._u_nj_m = zeros((self.total_samples, self.nstates_sampled))
        self._u_nj_k = zeros((self.total_samples, self.nstates_unsampled))
        mask0, maskn0 = self.mask_zero, self.mask_nonzero
        if _u_ijn.ndim == 2:
            for i, n in enumerate(self.nsamples_nonzero):
                j = self.nsamples_nonzero[:i].sum()
                self._u_nj_m[j:(j+n), :] = _u_ijn[maskn0, j:(j+n)].T
                self._u_nj_k[j:(j+n), :] = _u_ijn[mask0, j:(j+n)].T
        elif _u_ijn.ndim == 3:
            for i, n in enumerate(self.nsamples_nonzero):
                j = self.nsamples_nonzero[:i].sum()
                self._u_nj_m[j:(j+n), :] = _u_ijn[i, maskn0, :n].T
                self._u_nj_k[j:(j+n), :] = _u_ijn[i, mask0, :n].T
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

        FOR INTERNAL USE ONLY!
        """
        return hstack((self._u_nj_m, self._u_nj_k))

    @property
    def u_nj(self):
        """The reduced potential for all samples at all states."""
        u_nj = zeros((self.total_samples, self.nstates))
        u_nj[:, self.mask_nonzero] = self._u_nj_m
        u_nj[:, self.mask_zero] = self._u_nj_k
        return u_nj

    @property
    def _f(self):
        """The sample sorted reduced free energies in all states.

        FOR INTERNAL USE ONLY!
        """
        mask0, maskn0 = self.mask_zero, self.mask_nonzero
        _f = hstack((self.f[maskn0], self.f[mask0]))
        _f -= _f[0]
        return _f

    @property
    def _W_nj(self):
        """The sample sorted sample weights in all states.

        FOR INTERNAL USE ONLY!
        """
        m = self.nstates_sampled
        logQ_nj = self._f[newaxis, :] - self._u_nj
        logNorm_n = logsumexp(logQ_nj[:, :m], 1, self.PIsdiag[newaxis, :])
        _W_nj = exp(logQ_nj - logNorm_n[:, newaxis])
        return _W_nj

    @property
    def W_nj(self):
        """Compute the normalized sample weights for all samples in all states.
        
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

#    @property
#    def _R_nj(self):
#        R_nj = zeros((self.total_samples, self.nstates_sampled))
#        ncum = hstack((zeros(1, int32), self.nsamples_nonzero.cumsum()))
#        for j, (n1, n2) in enumerate(zip(ncum[:-1], ncum[1:])):
#            R_nj[n1:n2, j] += self.total_samples / (n2 - n1) 
#        return R_nj

    def _validate_and_convert_2d(self, A_jn):
        """Convenience function for accepting state/sample input in multiple
        shapes. Internally always use 1d arrays of size self.total_samples,
        but for computations it is often easier/more efficient to sort by
        state with a masked sample axis.
        """
        _A_jn = asarray(A_jn)
        if _A_jn.ndim == 1:
            A_n = A_jn
            assert A_n.size == self.total_samples
        elif _A_jn.ndim == 2:
            assert _A_jn.shape == (self.nstates_sampled, self.nsamples.max())
            if all(self.nsamples == self.nsamples[0]):
                # Shortcut for all elements containing data.
                A_n = A_jn.ravel()
            else:
                A_n = zeros(self.total_samples)
                for i, n in enumerate(self.nsamples_nonzero):
                    j = self.nsamples_nonzero[:i].sum()
                    A_n[j:(j+n)] = A_jn[i, :n]
        else:
            raise ValueError('Bad input shape')
        return A_n

    def compute_expectations(self, A_jn, doerror=True):
        """Compute the weighted expectations in each state. This assumes that
        the states here were all included in the initialization.

        Arguments
        ---------
        A_jn : array-like
            Either 1 or 2d array-like samples. 2d data is assumed to be sorted
            by state with the sample size as the input reduced potential
        doerror : boolean
            Compute variances if true

        Returns
        -------
        A_j : ndarray
            Expected values in each state
        varA_j : ndarray
            Estimated variance in each state (all zeros if doerror is False)
        """
        A_n = self._validate_and_convert_2d(A_jn)
        W_nj = self.W_nj
        A_j = (W_nj*A_n[:, newaxis]).mean(axis=0)
        varA_j = zeros(self.nstates)
        if not doerror:
            return A_j, varA_j

        """
        There are a number of errors in Ref 1. First, the definitions of W and
        WA are incorrect, the extra factors of exp(f) should indeed be
        included. Doing so obviates the need for the C matrix defined therein.
        This itself is used incorrectly in that paper since the dimensions are
        inconsistent during matrix multiplication.
        
        NB This is all borne out by R code released with Ref 1, which uses the
        same equations below, but with completely different notation (A1 --> G,
        B1 --> H). The matrix D in Ref 1 is NOT the same as in the code, where
        it seems to be the first term in the B matrix from the paper.
        
        Shorthand indices - notation similiar to Ref 1
        """
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
        """
        This is how things are computed in the R code. The solution of
        solve(B1, A1) is not numerically the same as B1invA1, although they are
        supposedly mathematically identical. Non-exhaustive tests show that
        these give virtually identical results for V, except for state 0.
        """
#        B1invA1 = (-O + tile(O[0], O[0].size).reshape(O.shape))[1:, 1:]
#        V = B1invA1.dot(inv(B1.T)) / n
        U = zeros((mpk2, mpk2))
        U[1:, 1:] = V
        Ch = hstack((diagflat(-_A_j), identity(mpk)))
        V_full = Ch.dot(U).dot(Ch.T)
        varA_j[maskn0] = diagonal(V_full)[:m]
        varA_j[mask0] = diagonal(V_full)[m:]
        return A_j, varA_j

    @property
    def fvar(self):
        """Estimate the variances of the reduced free energies."""
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
#            D = _W_nj.T.dot(self._R_nj) / n
#            A1 = (O - D.dot(self.PIs).dot(D.T))[1:, 1:]
#            V = solve(B1, A1).dot(inv(B1.T)) / n
        # Unshuffle the state indices. Note that the variance of state 0 is
        # zero by construction and thus is omitted from V - since the user
        # selected state 0 may be different from the lowest indexed sample
        # state, re-adjust so that the actual state 0 has zero variance.
        #
        V_full = zeros((mpk, mpk))
        V_full[1:, 1:] = V
        var = zeros(mpk)
        var[maskn0] += diagonal(V_full)[:m]
        var[mask0] += diagonal(V_full)[m:]
        var += var[0] 
        var[0] = 0.0
        return var

    def compute_unsampled_free_energies(self, u_jn, doerror=True):
        """Compute free energies in unsampled states. This requires the
        observed reduced potential energies on the complete sample set.

        NB Use of this function can be obviated by simply including the
        unsampled states in the initial calculation. 

        Arguments
        ---------
        u_jn : array-like
            2d array-like of reduced potential energies. The dimensions must
            be L x N, where L is the number of states to compute free energies
            for and N is the total sample size.
        doerror : boolean (optional)
            If true, also estimate variances

        Returns
        -------
        f_k : ndarray
            The free energies in the unsampled states
        varf_k : ndarray
            Estimated variance for each state (all zeros if doerror is False) 
        """
        u_nj_k = asarray(u_jn).T
        m, n = self.nstates_sampled, self.nsamples_nonzero
        logQ_nj = self._f[:m] - self._u_nj_m
        logw_n = -(u_nj_k + logsumexp(logQ_nj, 1, n[newaxis, :])[:, newaxis])
        f_k = -logsumexp(logw_n, axis=0)
        # Upon solution, the zero of energy is chosen _arbitrarily_, that is,
        # the shift is independent of the WHAM equations and so subsequent
        # free energies must be adjusted a posteriori.
        #
        try:
            f_k -= self._f0
        except AttributeError:
            pass
        varf_k = zeros(f_k.size)
        if not doerror:
            return f_k, varf_k

        # Shorthand indices - notation similiar to Ref. 1
        n, m, mpk = self.total_samples, self.nstates_sampled, self.nstates
        mask0, maskn0 = self.mask_zero, self.mask_nonzero
        newk = u_nj_k.shape[1]
        mpk += newk # expand k to include new states    
        k = mpk - m
        W_nj_k = self.compute_unsampled_weights(u_jn)
        _W_nj = hstack((self._W_nj, W_nj_k))
        O = _W_nj.T.dot(_W_nj) / n
        Os = O[:, :m]
        B1 = (hstack((Os.dot(self.PIs), zeros((mpk, k))))
              - identity(mpk))[1:, 1:]
        A1 = (O - Os.dot(self.PIs).dot(Os.T))[1:, 1:]
        V = solve(B1, A1).dot(inv(B1.T)) / n
        V_full = zeros((mpk, mpk))
        V_full[1:, 1:] = V
        var = zeros(mpk)
        var[maskn0] += diagonal(V_full)[:m]
        var[mask0] += diagonal(V_full)[m:mpk-newk]
        varf_k += diagonal(V_full)[mpk-newk:] + var[0]
        return f_k, varf_k

    def compute_unsampled_weights(self, u_jn):
        """Compute the sample weights for unsampled states. This requires the
        observed reduced potential energies on the complete sample set.

        NB Use of this function can be obviated by simply including the
        unsampled states in the initial calculation. 

        Arguments
        ---------
        u_jn : array-like
            2d array-like of reduced potential energies. The dimensions must
            be L x N, where L is the number of states to compute free energies
            for and N is the total sample size.

        Returns
        -------
        W_nj_k : ndarray
            Sample weights for the N samples in each of L states 
        """
        u_nj_k = asarray(u_jn).T
        # NB ESTIMATING ERRORS HERE WILL CAUSE AN INFINITE LOOP!
        f_k, varf_k = self.compute_unsampled_free_energies(u_jn, False)
        logQ_nj_k = f_k[newaxis, :] - u_nj_k
        m = self.nstates_sampled
        logQ_nj = self._f[newaxis, :] - self._u_nj
        logNorm_n = logsumexp(logQ_nj[:, :m], 1, self.PIsdiag[newaxis, :])
        return exp(logQ_nj_k - logNorm_n[:, newaxis])

    def compute_unsampled_expectations(self, A_jn, u_jn, doerror=True):
        """Compute expectations in unsampled states. This requires the
        observed reduced potential energies on the complete sample set.

        NB Use of this function can be obviated by simply including the
        unsampled states in the initial calculation. 

        Arguments
        ---------
        A_jn : array-like
            Either 1 or 2d array-like samples. 2d data is assumed to be sorted
            by state with the sample size as the input reduced potential
        u_jn : array-like
            2d array-like of reduced potential energies. The dimensions must
            be L x N, where L is the number of states to compute free energies
            for and N is the total sample size.
        doerror : boolean (optional)
            If true, also estimate variances

        Returns
        -------
        A_j : ndarray
            Expected values in each state
        varA_j : ndarray
            Estimated variance in each state (all zeros if doerror is False)
        """
        A_n = self._validate_and_convert_2d(A_jn)
        W_nj_k = self.compute_unsampled_weights(u_jn)
        A_j = (W_nj_k*A_n[:, newaxis]).mean(axis=0)
        varA_j = zeros(A_j.size)
        if not doerror:
            return A_j, varA_j

        # See notes in compute_expectations().
        n, m, k = self.total_samples, self.nstates_sampled, A_j.size
        mpk = m + k
        mpk2 = 2*mpk
        # Shuffle indices and re-define W (WWA, hereafter).
        W_nj_k = self.compute_unsampled_weights(u_jn)
        _W_nj = hstack((self._W_nj[:, :m], W_nj_k))
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
        U = zeros((mpk2, mpk2))
        U[1:, 1:] = V
        Ch = hstack((diagflat(-_A_j), identity(mpk)))
        V_full = Ch.dot(U).dot(Ch.T)
        varA_j += diagonal(V_full)[m:]
        return A_j, varA_j

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

        Arguments
        ---------
        f_guess : array-like
            Initial guess of the free energies.  If None, use all zeros.
        method : str
            Minimization method.  This is passed directly to
            scipy.optimize.minimize.  See the scipy documentation for details.
        tol : float, optional
            Tolerance for termination (see scipy.optimize.minimize).

        Returns
        -------
        sltn : OptimizeResult (see scipy.optimize.minimize)
        """
        _f_guess = self._objective_f(f_guess)
        jac_methods = ['CG', 'BFGS', 'NEWTON-CG', 'L-BFGS-B', 'TNC', 'SLSQP',
                       'DOGLEG', 'TRUST-NCG']
        hess_methods = ['NEWTON-CG', 'DOGLEG', 'TRUST-NCG']
        _method = str(method).upper()

        _jac = (_method in jac_methods)
        _hess = (self._uwham_hess if _method in hess_methods else None)
        sltn = minimize(self._uwham_obj_grad, _f_guess, method=_method, 
                        jac=_jac, hess=_hess, tol=float(tol), 
                        options={'disp': verbose})
        #   Now estimate the unsampled state free energies and map everything 
        # back to the original indices. Note that the lowest index of the 
        # sampled states was constrained to zero.
        #
        _f = hstack((zeros(1), sltn.x))
        self.f = zeros(self.nstates)
        self.f[self.mask_nonzero] += _f 
        f_k, tmp = self.compute_unsampled_free_energies(self._u_nj_k.T, False)
        self.f[self.mask_zero] += f_k
        self._f0 = self.f[0] # Store in case of additional free energies
        self.f -= self._f0
        self.check_sample_weight_normalization()
        return sltn 

    def check_sample_weight_normalization(self, tol=1e-6):
        """Return False if normalization criteria are not met, else True.

        Arguments
        ---------
        tol : float
            Tolerance for absolute error to vanish
        """
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

        Arguments
        ---------
        f_i : array-like (optional)
            Free energies to test at. If None, use all zeros.
        eps : float
            Displacement value for finite differencing.

        Returns
        -------
        grad_anal : ndarray
            Analytic gradient
        grad_numer : ndarray
            Numerical gradient
        hess_anal : ndarray
            Analytic Hessian
        hess_numer : ndarray
            Numerical Hessian
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

