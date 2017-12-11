from __future__ import division
import collections
import json
import warnings

import numpy as np
from scipy.optimize import root

from pynamd.msmle import MSMLE


# It might be useful to implement our own root finding algorithm and avoid
# dependency on scipy.optimize...
#def _bisect_vector(f, a, b, tol=1e-5):
#    """Vector version of bisection method for root finding."""
#    xmid = 0.5*(a + b)
#    fa = f(a)
#    fb = f(b)
#    fmid = f(xmid)
#    a = np.where(np.sign(fmid) == np.sign(fa), xmid, a)
#    b = np.where(np.sign(fmid) == np.sign(fb), xmid, b)
#    maxerr = np.amax(np.abs(b - a))
#    if maxerr < tol:
#        return xmid
#    else:
#        return _bisect_vector(f, a, b, tol)

def _validate_typed_list(list_, type_):
    """Check that the argument can be cast as a list of a single type."""
    if not hasattr(list_, '__iter__'):
        raise ValueError('lists must be iterable')
    for i, v in enumerate(list_):
        try:
            type_(v)
        except ValueError:
            raise ValueError(
                    'Cannot cast list element %s as %s'%(str(v), str(type_))
                  )
        list_[i] = type_(v)
    return list_

def _validate_state_dict(dict_):
    """Check that the argument is a dictionary where all keys are strings and
    all arguments are equal length lists of integers.
    """
    if not isinstance(dict_, dict):
        raise ValueError('states must be a dict object')
    for k, v in dict_.iteritems():
        dict_.pop(k, None)
        try:
            dict_[str(k)] = _validate_typed_list(v, int)
        except ValueError:
            raise ValueError('Values in state dicts must be integer lists!')
    vlen = len(dict_.itervalues().next())
    for k, v in dict_.iteritems():
        if len(v) == vlen:
            continue
        raise ValueError(
                'Bad list length for state %s (%d != %d)'%(k, len(v), vlen)
              )
    return dict_

def _validate_float(value):
    try:
        float(value)
    except ValueError:
        raise ValueError('Value must be a float quantity')
    return float(value)


class TitratableSystemSet(collections.Mapping):
    """A dict-like object containing multiple TitratableSystems sorted by pH.
    """
    # These should always be compared with str(method).lower()
    _MULTISTATE_METHODS = ('uwham')

    def __init__(self, *args, **kwargs):
        self._od = collections.OrderedDict(*args, **kwargs)

    def __delitem__(self, pH):
        del self._od[_validate_float(pH)]

    def __getitem__(self, pH):
        return self._od[_validate_float(pH)]

    def __iter__(self):
        return iter(self._od)

    def __len__(self):
        return len(self._od)

    def __setitem__(self, pH, system):
        """Adds restriction that all keys be pH values and all values be
        TitratableSystem objects.
        """
        if not isinstance(system, TitratableSystem):
            raise ValueError(
              'TitratableSystemSets can only contain TitratableSystem objects'
            )
        # TODO: Enforce that the systems have the same residue structure.
        self._od[_validate_float(pH)] = system

    @property
    def pHs(self):
        """The pH values at which data was collected."""
        return np.asarray(self.keys())

    @property
    def numpHs(self):
        """The number of pH values (or systems)."""
        return self.pHs.size

    @property
    def nsamples(self):
        """The number of samples at each pH as an ndarray."""
        return np.asarray([s.nsamples for s in self.itervalues()], np.int32)

    @property
    def nsites(self):
        """The number of protonation sites per residue (same at all pH values)
        """
        return self.values()[0].nsites

    @property
    def nprotons(self):
        nprotons = np.zeros(self.nsamples.sum(), np.int32)
        indices = np.hstack((np.zeros(1, np.int32), self.nsamples.cumsum()))
        for i, j, tsys in zip(indices[:-1], indices[1:], self.itervalues()):
            nprotons[i:j] += tsys.nprotons
        return nprotons

    @property
    def nstates_micro_noequiv(self):
        """The number of microstates per residue (same at all pH values)"""
        return self.values()[0].nstates_micro_noequiv

    @property
    def nstates_micro_equiv(self):
        """The number of microstates per residue (same at all pH values)"""
        return self.values()[0].nstates_micro_equiv

    @property
    def nstates_macro(self):
        """The number of macrostates per residue (same at all pH values)"""
        return self.values()[0].nstates_macro

    def nresidues(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """Return the number of residues in the given selection. This is the
        same at all pH values.

        (Optional) selection keywords
        -----------------------------
        segresids : list
            explicit residue selections of the form <segid:resid>
        notsegresids : list
            explicit residue exclusions of the form <segid:resid>
        resnames : list
            explict selection by residue name
        notresnames : list
            explicit exclusion by residue name
        """
        args = (segresids, notsegresids, resnames, notresnames)
        return self.values()[0].nresidues(*args)

    def segresids(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """Return, as a list, the segresids in the given selection. This is the
        same at all pH values.

        (Optional) selection keywords
        -----------------------------
        segresids : list
            explicit residue selections of the form <segid:resid>
        notsegresids : list
            explicit residue exclusions of the form <segid:resid>
        resnames : list
            explict selection by residue name
        notresnames : list
            explicit exclusion by residue name
        """
        args = (segresids, notsegresids, resnames, notresnames)
        return self.values()[0].segresids(*args)

    def resnames(self, segresids=[], notsegresids=[], resnames=[], 
            notresnames=[]):
        """Return, as a list, the resnames in the given selection. This is the
        same at all pH values.

        (Optional) selection keywords
        -----------------------------
        segresids : list
            explicit residue selections of the form <segid:resid>
        notsegresids : list
            explicit residue exclusions of the form <segid:resid>
        resnames : list
            explict selection by residue name
        notresnames : list
            explicit exclusion by residue name
        """
        args = (segresids, notsegresids, resnames, notresnames)
        return self.values()[0].resnames(*args)

    def _combine_occupancies(self, nstates, occupancy_type, segresids,
            notsegresids, resnames, notresnames):
        # Allocate an array large enough for all residues at all pH values.
        # This may also include masking for residue selections. Stack all of
        # the data in this array.
        #
        args = (segresids, notsegresids, resnames, notresnames)
        mask = self.values()[0]._selection_mask(*args)
        _nstates = nstates*mask
        occ = np.zeros((self.nsamples.sum(), _nstates.sum()), np.int32)
        indices = np.hstack((np.zeros(1, np.int32), self.nsamples.cumsum()))
        for i, j, tsys in zip(indices[:-1], indices[1:], self.itervalues()):
            occ[i:j] += tsys.__getattribute__(occupancy_type)(*args)
        return occ

    def micro_occupancies_noequiv(self, segresids=[], notsegresids=[],
            resnames=[], notresnames=[]):
        """Return the microstate occupancies (with no adherence to equivalent
        states) from each pH value stacked as a ndarray.

        (Optional) selection keywords
        -----------------------------
        segresids : list
            explicit residue selections of the form <segid:resid>
        notsegresids : list
            explicit residue exclusions of the form <segid:resid>
        resnames : list
            explict selection by residue name
        notresnames : list
            explicit exclusion by residue name

        Returns
        -------
        occupancy : ndarray
            the occupancies of the selected residues
        """
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nstates_micro_noequiv
        otype = 'micro_occupancies_noequiv'
        return self._combine_occupancies(nstates, otype, *args)

    def micro_occupancies_equiv(self, segresids=[], notsegresids=[],
            resnames=[], notresnames=[]):
        """Return the microstate occupancies (combining equivalent states) from
        each pH value stacked as a ndarray.

        (Optional) selection keywords
        -----------------------------
        segresids : list
            explicit residue selections of the form <segid:resid>
        notsegresids : list
            explicit residue exclusions of the form <segid:resid>
        resnames : list
            explict selection by residue name
        notresnames : list
            explicit exclusion by residue name

        Returns
        -------
        occupancy : ndarray
            the occupancies of the selected residues
        """
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nstates_micro_equiv
        otype = 'micro_occupancies_equiv'
        return self._combine_occupancies(nstates, otype, *args)

    def macro_occupancies(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """Return the macrostate occupancies from each pH value stacked as a
        ndarray.

        (Optional) selection keywords
        -----------------------------
        segresids : list
            explicit residue selections of the form <segid:resid>
        notsegresids : list
            explicit residue exclusions of the form <segid:resid>
        resnames : list
            explict selection by residue name
        notresnames : list
            explicit exclusion by residue name

        Returns
        -------
        occupancy : nstates ndarray
            the occupancies of the selected residues
        """
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nstates_macro
        otype = 'macro_occupancies'
        return self._combine_occupancies(nstates, otype, *args)

    def site_occupancies(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nsites
        otype = 'site_occupancies'
        return self._combine_occupancies(nstates, otype, *args)

    @classmethod
    def from_cphlogs(cls, cphlogs, configfiles, start=None, stop=None,
                     step=None):
        """Create a TitratableSystemSet from a list of cphlog files.

        Arguments
        ---------
        cphlogs : iterable
            A set of one or more cphlog filenames to be read
        configfiles : iterable
            A set of one or more constant pH configuration files to be read
        start : int 
            Subsampling of the aggregate data at each pH starts at this step.
        stop : int
            Subsampling of the aggregate data at each pH stops at this step.
        step : int
            Interval at which subsampling occurs 
        """
        # Make a temporary OrderedDict so that we can sort the pH values.
        tmp = collections.OrderedDict()
        for cphlog in cphlogs:
            tsys = TitratableSystem.from_cphlog(cphlog, configfiles)
            if tsys.pH not in tmp:
                tmp[tsys.pH] = tsys
            else:
                tmp[tsys.pH] += tsys
        pHs = tmp.keys()
        pHs.sort()
        obj = cls()
        for pH in pHs:
            tmp[pH].subsample(start, stop, step)
            obj[pH] = tmp[pH]
        del tmp
        return obj

    def compute_Hill_fit(self, segresid, micro=False, noequiv=False,
            est_method='uwham', **kwopts):
        """Compute the apparent pKa and Hill coefficient.

        This is NOT a non-linear regression. Instead the titration curve(s)
        is/are computed as numerical functions via WHAM and then the
        appropriate root is found to satisfy the Hill equation (i.e. linear
        dependence of the pKa on pH).
        """
        if micro:
            if noequiv:
                occs = self.micro_occupancies_noequiv([segresid]).T
            else:
                occs = self.micro_occupancies_equiv([segresid]).T
        else:
            occs = self.macro_occupancies([segresid]).T

        # Compute the WHAM weights if we haven't already for some reason.
        try:
            self.msmle
        except AttributeError:
            self._compute_multistate_weights(est_method, **kwopts)
        nstates = occs.shape[0]
        nprotons = self.nprotons

        log10 = np.log(10)
        def compute_mean(A, pH):
            # Convenience function/shorthand
            # Use WHAM weights to interpolate/extrapolate a mean at the given
            # pH value. For simplicity, ignore error analysis and assume that
            # we only use ONE pH at a time.
            u = log10*pH*nprotons[np.newaxis, :]          
            return self.msmle.compute_unsampled_expectations(A, u, False)[0][0]

#        def compute_err(A, pH):
#            u = log10*pH*nprotons[np.newaxis, :]
#            return np.sqrt(
#                self.msmle.compute_unsampled_expectations(A, u, True)[1][0]
#            )
        # Here we define different functions depending on the case.
        #
        # tcurve : return all titration curves at the given pH
        #   pH : float
        #   titration_curve : 1d ndarray, size = nstates
        #
        # obj : the objective function whose root is the apparent pKa.
        #   pH  : 1d ndarray, size = nstates
        #   obj : 1d ndarray, size = nstates
        # 
        # hill - return the Hill coefficients from the covariances at the pKas
        #   pKas : 1d ndarray, size = nstates
        #   hillcoeffs : 1d ndarray, size = nstates
        #
        if nstates == 1:
            def tcurve(pH):
                return np.array([compute_mean(occs[0], pH)])

            def obj(pH):
                return tcurve(pH) - 0.5

            def hill(pKa):
                nprot_p = compute_mean(nprotons*occs[0], pKa)
                nprot = compute_mean(nprotons, pKa)
                p = tcurve(pKa)[0]
                return np.array([4*(nprot_p - p*nprot)])
        elif nstates == 2:
            if micro:
                # Microscopic states with a shared endpoint
                # e.g. histidine or asymmetric carboxylates
                # NB: sign slip when shared state is [0, 0] or [1, 1]
                pKaM, nM = self.compute_Hill_fit(segresid, False, False,
                        est_method, **kwopts)
                missing_prot_cnts =\
                        self.values()[0][segresid].missing_proton_counts 
                if 0 in missing_prot_cnts: # common endpoint is (1, 1)
                    fac = -1.0
                elif 2 in missing_prot_cnts: # common endpoint is (0, 0)
                    fac = 1.0
                else:
                    raise ValueError("Badly constructed 2 state residue...")
                f = lambda x: 1 / (1 + 10**(fac*nM[0]*(pKaM[0] - x)))

                def tcurve(pH):
                    return np.array([compute_mean(occ, pH) for occ in occs])

                def obj(pH):
                    obj1 = (tcurve(pH[0])[0] - f(pH[0]))
                    obj2 = (tcurve(pH[1])[1] - f(pH[1]))
                    return np.asarray([obj1, obj2])

                def hill(pKas):
                    hillcoeffs = np.zeros(2)

                    nprot1 = compute_mean(nprotons, pKas[0])
                    p1 = tcurve(pKas[0])
                    nprot_p11 = compute_mean(nprotons*occs[0], pKas[0])
                    nprot_p21 = compute_mean(nprotons*occs[1], pKas[0])
                    cov11 = nprot_p11 - p1[0]*nprot1
                    cov21 = nprot_p21 - p1[1]*nprot1
                    hillcoeffs[0] = fac*(cov21 + 2*cov11) / f(pKas[0])
                            
                    nprot2 = compute_mean(nprotons, pKas[1])
                    p2 = tcurve(pKas[1])
                    nprot_p12 = compute_mean(nprotons*occs[0], pKas[1])
                    nprot_p22 = compute_mean(nprotons*occs[1], pKas[1])
                    cov12 = nprot_p12 - p2[0]*nprot2
                    cov22 = nprot_p22 - p2[1]*nprot2
                    hillcoeffs[1] = fac*(cov12 + 2*cov22) / f(pKas[1])
                    return hillcoeffs
            else:
                # diprotic residue
                raise ValueError('Not implemented')
        else:
            raise ValueError('Not implemented')

        # Solve the apparent pKa as the root of the objective function.
        soltn = root(obj, np.tile(self.pHs.mean(), nstates))
        apparent_pKas = soltn.x
        hillcoeffs = hill(apparent_pKas)
        hillcoeffs[apparent_pKas < 0] = np.nan
        apparent_pKas[apparent_pKas < 0] = np.nan

#        sites = self.macro_occupancies() #self.site_occupancies()
#        hill_psite = np.zeros(sites.shape[1])
#        for i, site in enumerate(sites.T):
#            s = compute_mean(site, u, False)[0][0]
#            sp_cov = compute_mean(site*occ, u, False)[0][0]
#            hill_psite[i] += 4*(sp_cov - pfrac*s)
#        print hill_psite
#        print hill_psite.sum()
        return apparent_pKas, hillcoeffs

    def compute_titration_curves(self, segresids=[], notsegresids=[],
            resnames=[], notresnames=[], micro=False, noequiv=False,
            est_method='uwham', z=1.0, **kwopts):
        """Compute titration curves for some (or all) of the system.

        Any number of residues can be selected specifically, and the size of
        the output will vary accordingly. Selections can be made by residue as
        <segid:resid> or by name. If no selections are made, then the entire
        system is returned.

        Several kinds of output are possible depending on optional keywords.
        By default, all residues are treated macroscopically, as if they were
        independently isolated. If micro evaluates True, then the microscopic
        states of each residue are treated separately (the macroscopic states
        are an aggregrate or complement of these states). By default, 
        chemically indistinct (i.e., equivalent) states are not separated 
        (e.g., the protons on carboxylic acids). These states are treated
        separately if noequiv evaluates True. Note that, in many cases, such
        states may not be obviously statistically identical, even within
        reasonable error estimates.

        Optional Keyword Arguments
        --------------------------
        segresids : list
            Explicit residue selections of the form <segid:resid>
        notsegresids : list
            Explicit residue exclusions of the form <segid:resid>
        resnames : list
            Explict selection by residue name
        notresnames : list
            Explicit exclusion by residue name
        micro : bool (default: False)
            If true, analyze microscopic states within a residue separately,
            otherwise treat the residue macroscopically.
        noequiv : bool (default: False)
            If True, analyze non-unique states separately, otherwise aggregate
            those states such that all states are unique.
        est_method : str (default: 'uwham')
            Estimation method for state populations. Supported options are
            'uwham', 'wald', 'yates', and 'agresti-coull'
        z : float (default: 1.0)        
            Confidence interval parameter (i.e. number of standard deviations
            of the mean). This may affect the extent of bias in non-multi-state
            estimators.

        Additional keyword options not listed here are passed directly to the
        MSMLE solver (see msmle documentation for details).

        Returns
        -------
        tcurve_dict : OrderedDict

        titration_curves : ndarray
            Two dimensional array of all requested titration curves
        titration_curve_errs : ndarray
            Two dimensional array of standard error estimates for all of the
            requested titration curves
        """
        est_method = str(est_method).lower()
        z2 = float(z)**2 

        maskargs = (segresids, notsegresids, resnames, notresnames)
        mask = self.values()[0]._selection_mask(*maskargs)
        if micro:
            if noequiv:
                occs = self.micro_occupancies_noequiv(*maskargs)
                nstates = mask*self.nstates_micro_noequiv
            else:
                occs = self.micro_occupancies_equiv(*maskargs)
                nstates = mask*self.nstates_micro_equiv
        else:
            occs = self.macro_occupancies(*maskargs)
            nstates = self.nstates_macro
        titration_curves = np.zeros((nstates.sum(), self.numpHs))
        titration_curve_errs = np.zeros((nstates.sum(), self.numpHs))

        if est_method in self._MULTISTATE_METHODS:
            self._compute_multistate_weights(est_method, **kwopts)
            warnings.simplefilter("error", RuntimeWarning)
            # Iterate each state and compute the titration curve at all pHs.
            for i, occ in enumerate(occs.T):
                p, pvar = self.msmle.compute_expectations(occ, True)
                titration_curves[i] += p
                try:
                    titration_curve_errs[i] += np.sqrt(z2*pvar)
                except RuntimeWarning:
                    titration_curve_errs[i] = np.nan
        else:
            if est_method in ('wald', 'naive', 'yates'):
                def get_pop(occ):
                    # The Yates method essentially adds a correction to the
                    # standard Wald estimate and then re-scales it. It is also
                    # sometimes called the Wilson score interval or just the
                    # Wilson interval.
                    n, p = occ.shape[0], occ.mean(axis=0)
                    if est_method == 'yates':
                        p = (2*n*p + z2) / (2*(n + z2))
                    q = 1 - p
                    pvar = z2*p*q / n
                    return p, pvar
            elif est_method == 'agresti-coull':
                def get_pop(occ):
                    # The Agresti-Coull method essentially redefines the sample
                    # size and shifts the data.
                    n = occ.shape[0] + z2
                    p = (occ.sum(axis=0) + 0.5*z2) / n
                    q = 1 - p
                    pvar = z2*p*q / n
                    return p, pvar
            else:
                raise ValueError('Unrecognized method %s'%est_method)
            # Iterate each pH and compute the titration curve for each state.
            _0 = np.zeros(1, np.int32) 
            indices = np.hstack((_0, self.nsamples.cumsum()))
            for j, (n1, n2) in enumerate(zip(indices[:-1], indices[1:])):
                p, pvar = get_pop(occs[n1:n2])
                titration_curves[:, j] += p
                titration_curve_errs[:, j] += np.sqrt(pvar)
        # Now package the titration curves as an ordered dict and separate by
        # residue. Use the combined segresidname as keys.
        #
        segresids = self.segresids(*maskargs)
        resnames = self.resnames(*maskargs)
        nstates = nstates[nstates > 0]
        tcurve_dict = collections.OrderedDict()
        i = 0
        for n, segresid, resname in zip(nstates, segresids, resnames):
            segresidname = '%s:%s'%(segresid, resname)
            j = i + n
            tcurve_dict[segresidname] = \
                    (titration_curves[i:j], titration_curve_errs[i:j])
            i = j
        return tcurve_dict 

    def _compute_multistate_weights(self, est_method, **kwopts):
        """Create an MSMLE object with the given parameters."""
        _method = str(est_method).lower()
        if _method not in self._MULTISTATE_METHODS:
            raise ValueError('Unrecognized MSMLE method %s'%str(est_method))

        # TODO: permit estimation at unsampled pH values.
        u = np.zeros((self.numpHs, self.numpHs, self.nsamples.max()))
        log10 = np.log(10)
        pHs = self.pHs
        for k, (tsys, n) in enumerate(zip(self.itervalues(), self.nsamples)):
            u[k, :, :n] = log10*pHs[:, np.newaxis]*tsys.nprotons
        msmle = MSMLE(u, self.nsamples)

        # TODO: permit other estimation methods as they become available. 
        if _method == 'uwham':
            msmle.solve_uwham(**kwopts)
        self.msmle = msmle


class TitratableSystem(collections.Mapping):
    """A system of multiple titratable residues in contact with the same pH

    Parameters
    ----------
    pH : float
        The pH value that the residues are in contact with
    titratable_residues : TitratableResidue
        One or more TitratableResidue objects
    """
    def __init__(self, pH, *args, **kwargs):
        self.pH = _validate_float(pH)
        self._od = collections.OrderedDict(*args, **kwargs)

    def __delitem__(self, segresid):
        del self._od[str(segresid)]

    def __getitem__(self, segresid):
        return self._od[str(segresid)]

    def __iter__(self):
        return iter(self._od)

    def __len__(self):
        return len(self._od)

    def __setitem__(self, segresid, titratable_residue):
        """Adds restriction that all keys be <segid:resid> labels and all
        values be TitratableResidue objects.
        """
        tokens = str(segresid).split(':')
        if len(tokens) != 2:
            raise ValueError('segresid must be of the form <segid:resid>')
        try:
            int(tokens[1])
        except ValueError:
            raise ValueError('resid must be an integer, got %s'%tokens[1])
        if not isinstance(titratable_residue, TitratableResidue):
            raise ValueError(
              'TitratableSystems can only contain TitratableResidue objects'
            )
        # TODO: Enforce that the systems have the same residue structure.
        self._od[str(segresid)] = titratable_residue

    def subsample(self, start, stop, step):
        """Modify the data set for all residues in-place."""
        for tres in self.itervalues():
            tres.site_occupancies = tres.site_occupancies[start:stop:step]

    @property
    def nresidues(self):
        """The number of residues in the system"""
        return len(self)

    @property
    def nprotons(self):
        """The time series of number of protons in the system

        This is just the sum of site occupancies of all residues.
        """
        return np.hstack(
            (tres.site_occupancies for tres in self.itervalues())
        ).sum(axis=1)

    @property
    def nsites(self):
        """The number of sites per residue"""
        return np.array([tres.nsites for tres in self.itervalues()], np.int32)

    @property
    def nsamples(self):
        """The number of occupation vector samples"""
        nsamples = self.values()[0].nsamples
        assert np.all(
          np.asarray([tres.nsamples for tres in self.itervalues()]) == nsamples
        )
        return nsamples

    @property
    def nstates_micro_noequiv(self):
        """The number of microstates per residue without equivalencing"""
        nstates = [tres.nstates_micro_noequiv for tres in self.itervalues()]
        return np.array(nstates, np.int32)

    @property
    def nstates_micro_equiv(self):
        """The number of microstates per all residue with equivalencing"""
        nstates = [tres.nstates_micro_equiv for tres in self.itervalues()]
        return np.array(nstates, np.int32)

    @property
    def nstates_macro(self):
        """The number of macrostates per residues"""
        nstates = [tres.nstates_macro for tres in self.itervalues()]
        return np.array(nstates, np.int32)

    def _selection_mask(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        # Return a boolean array indicating the residues to be analyzed
        # TODO: Do any kind of error checking here...
        mask = np.zeros(len(self), np.int32)
        if ((len(segresids) == len(notsegresids) == 0)
            and (len(resnames) == len(notresnames) == 0)):
            mask += 1 
            return mask
        if len(notsegresids) > 0 or len(notresnames) > 0:
            mask += 1
            for i, tres in enumerate(self.itervalues()):
                if (tres.segresid in notsegresids
                    or tres.resname in notresnames):
                    mask[i] = 0
        for i, tres in enumerate(self.itervalues()):
            if tres.segresid in segresids or tres.resname in resnames:
                mask[i] = 1
        return mask

    def _get_masked_occupancy(self, nstates, occupancy_type, mask):
        # Allocate an array for just those protonation states that have been
        # selected. Since we have to iterate the TitratableResidue objects
        # anyway, rather than do a numpy mask just zero out the state count
        # neglect those states when allocating the full array.
        #
        # This is probably horribly fragile if not used carefully...
        _nstates = mask*nstates
        occ = np.zeros((self.nsamples, _nstates.sum()), np.int32)
        i = 0
        for tres, n in zip(self.itervalues(), _nstates):
            if n == 0:
                continue
            j = i + n
            occ[:, i:j] += tres.__getattribute__(occupancy_type)
            i = j
        return occ

    def nresidues(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """See TitratableSystemSet.nresidues."""
        args = (segresids, notsegresids, resnames, notresnames)
        return self._selection_mask(*args).sum()

    def segresids(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """See TitratableSystemSet.segresids."""
        args = (segresids, notsegresids, resnames, notresnames)
        mask = self._selection_mask(*args)
        return [tres.segresid for tres, m in zip(self.itervalues(), mask) if m]

    def resnames(self, segresids=[], notsegresids=[], resnames=[], 
            notresnames=[]):
        """See TitratableSystemSet.segresids."""
        args = (segresids, notsegresids, resnames, notresnames)
        mask = self._selection_mask(*args)
        return [tres.resname for tres, m in zip(self.itervalues(), mask) if m]
 
    def micro_occupancies_noequiv(self, segresids=[], notsegresids=[], 
            resnames=[], notresnames=[]):
        """See TitratableSystemSet.micro_occupancies_noequiv."""
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nstates_micro_noequiv
        otype = 'micro_occupancies_noequiv'
        mask = self._selection_mask(*args)
        return self._get_masked_occupancy(nstates, otype, mask)

    def micro_occupancies_equiv(self, segresids=[], notsegresids=[],
            resnames=[], notresnames=[]):
        """See TitratableSystemSet.micro_occupancies_equiv."""
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nstates_micro_equiv
        otype = 'micro_occupancies_equiv'
        mask = self._selection_mask(*args)
        return self._get_masked_occupancy(nstates, otype, mask)

    def macro_occupancies(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """See TitratableSystemSet.macro_occupancies."""
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nstates_macro
        otype = 'macro_occupancies'
        mask = self._selection_mask(*args)
        return self._get_masked_occupancy(nstates, otype, mask) 
 
    def site_occupancies(self, segresids=[], notsegresids=[], resnames=[],
            notresnames=[]):
        """See TitratableSystemSet.site_occupancies."""
        args = (segresids, notsegresids, resnames, notresnames)
        nstates = self.nsites
        otype = 'site_occupancies'
        mask = self._selection_mask(*args)
        return self._get_masked_occupancy(nstates, otype, mask)

    def __repr__(self):
        return 'TitratableSystem(%s, %s)'%(str(self.pH), list.__repr__(self))

    def __eq__(self, other):
        if (self.pH != other.pH) or (len(self) != len(other)):
            return False
        for (sres, ores) in zip(self.iteritems(), other.iteritems()):
            if sres[0] != ores[0] or sres[1].resname != ores[1].resname:
                return False
        return True

    def __iadd__(self, other):
        """In-place addition - concatenate occupancies if otherwise equal."""
        if not self == other:
            raise TypeError('Cannot add - mismatch in system pH or residues.')
        for (sres, ores) in zip(self.itervalues(), other.itervalues()):
            sres += ores
        return self

    @staticmethod
    def _read_cphlog(cphlog):
        """Read a cphlog and return the header info (i.e. pH and residue info)
        as well as the occupancy as a numpy array.

        Returns:
        --------
        pH : float
          The pH value at which the data was generated
        res_strs : list
          A list of <segid:resid:resname> strings for titratable residue
        occupancy : 2d-ndarray (dtype=int32)
          The occupancy vectors at each cycle with shape (N, D) for N cycles
          and D sites in the system
        """
        pH = None
        res_strs = None
        # Read just the header information, designated as all comment lines
        # (prefaced with a '#') until the first non-comment line.
        for line in open(cphlog, 'r'):
            if not line.startswith('#'):
                break
            tokens = line.lstrip('#').strip().split()
            try:
                # Look for system info (i.e. pH).
                pH = float(tokens[tokens.index('pH')+1])
            except ValueError:
                # Otherwise look for a residue list.
                res_strs = tokens
        if pH is None:
            raise ValueError('No pH info in header of %s'%cphlog)
        if res_strs is None:
            raise ValueError('No residue info in header of %s'%cphlog)
        # Read the occupancy trajectory (first column is cycle number).
        occupancy = np.loadtxt(cphlog, np.int32)[:, 1:]
        return (pH, res_strs, occupancy)

    @classmethod
    def _from_cphlog(cls, cphlog, json_data):
        """Create a TitratableSystem object from a single cphlog.

        This is meant to be a convenience function for generating multiple
        TitratableSystem objects for in-place addition. This permits use of the
        error checking from __iadd__().
        """
        pH, segresidnames, occ = TitratableSystem._read_cphlog(cphlog)
        obj = cls(pH)
        i = 0
        for segresidname in segresidnames:
            segid, resid, resname = segresidname.split(':')
            states = json_data[resname]['states']
            pKas = json_data[resname]['pKa']
            nsites = len(states.itervalues().next())
            j = i + nsites
            segresid = '%s:%s'%(segid, resid)
            obj[segresid] =\
                    TitratableResidue(segresidname, states, pKas, occ[:, i:j])
            i = j
        return obj

    @staticmethod
    def read_json(*configfiles):
        """Read template info from one or more json configuration files."""
        # TODO: warn the user when a resname is re-defined?
        json_data = {}
        for configfile in configfiles:
            json_data.update(json.load(open(configfile, 'r')))
        return json_data

    @classmethod
    def from_cphlog(cls, cphlogs, configfiles):
        """Create a TitratableSystem object from one or more cphlogs (as a 
        list) using one or more JSON configfiles (as a list).
        """
        json_data = TitratableSystem.read_json(*configfiles)
        if not hasattr(cphlogs, '__iter__'):
            return cls._from_cphlog(cphlogs, json_data)
        obj = cls._from_cphlog(cphlogs[0], json_data)
        for cphlog in cphlogs[1:]:
            obj += cls._from_cphlog(cphlog, json_data)
        return obj


class TitratableResidue(object):
    """A residue object for a specific titratable residue in a system.

    Parameters
    ----------
    segresidname : str
        Unique identifier of the form <segid:resid:resname>
    states : dict
        Dictionary with keys for each state and values for occupancy vectors
    pKas : iterable
        pKa values for each state (must match values in states)
    occupancies : array-like (can be multiple)
        One or more arrays containing time series of the occupancy vector. 
        Should have shape (N, D) for N timesteps with D sites. For single site
        trajectories shape (N,) is also permissible.
    """
    def __init__(self, segresidname, states, pKas, *occupancies):
        tokens = str(segresidname).split(':')
        if len(tokens) != 3 or not isinstance(int(tokens[1]), int):
            raise ValueError(
                    'segresidname must be of form <segid:resid:resname>'
            )
        self.segresid = (':'.join(tokens[0:2])).upper()
        self.resname = tokens[2].upper()
        self.states = _validate_state_dict(states)
        self.pKas = np.asarray(pKas, np.float64)
        # Check that occupancies:
        #  1) match the dimensions in self.states
        #  2) are internally consistent (evidenced by broadcast errors)
        #
        self.site_occupancies = np.atleast_2d(
                np.asarray(occupancies[0], np.int32)
        )
        obs_nsites = self.site_occupancies.shape[1]
        if self.nsites != obs_nsites:
            raise ValueError(
                    'mismatch in template and observed occupancies'
                     ' (%d != %d)'%(self.nsites, obs_nsites)
            )
        # Use in-place addition to recycle shape error checking.
        for occ in occupancies[1:]:
            self += TitratableResidue(states, pKas, occ)
    
    @property
    def nsites(self):
        """The number of (possibly non-unique) sites in this residue"""
        return len(self.states.itervalues().next())

    @property
    def nsamples(self):
        """The number of observed occupation vectors (i.e., samples)"""
        return self.site_occupancies.shape[0]

    @property
    def nstates(self):
        """The number states this residue can have"""
        return len(self.states)

    @property
    def nstates_micro_noequiv(self):
        """The number of microstates when not accounting for equivalent states
        """
        return self.micro_occupancies_noequiv.shape[1]

    @property
    def nstates_micro_equiv(self):
        """The number of microstates when accounting for equivalent states
        """
        return self.micro_occupancies_equiv.shape[1]

    @property
    def nstates_macro(self):
        """The number of macrostates"""
        return self.macro_occupancies.shape[1]

    @property
    def missing_proton_counts(self):
        """A list of proton counts that are NOT attained by any state

        NB: The max # of protons is self.nsites and the min is zero.
        """
        proton_count_exists = [0 for i in xrange(self.nsites + 1)]
        for occ in self.states.itervalues():
            proton_count_exists[sum(occ)] = 1
        _missing_proton_counts = []
        for i, exists in enumerate(proton_count_exists):
            if not exists:
                _missing_proton_counts.append(i)
        return _missing_proton_counts

    def __repr__(self):
        segresidname = ':'.join([self.segresid, self.resname])
        return ('TitratableResidue(%s, %s, %s)'
                %(segresidname, str(self.states), str(self.pKas))
               )

    def __eq__(self, other):
        if not (self.pKas == other.pKas).all() or self.states != other.states:
            return False
        return True

    def __iadd__(self, other):
        """In-place addition - concatenate occupancies if otherwise equal."""
        if not self == other:
            raise TypeError('Cannot add - mismatch in residue properties.')
        if self.site_occupancies.shape[1] != other.site_occupancies.shape[1]:
            raise ValueError(
                    ('Cannot add - mismatch in occupancy shapes. %s != %s'
                     %(str(self.site_occupancies.shape[1]),
                       str(other.site_occupancies.shape[1]))
                    )
            )
        self.site_occupancies = np.concatenate(
                (self.site_occupancies, other.site_occupancies)
        )
        return self

    @property
    def micro_occupancies_noequiv(self):
        """State occupancies for this residue with no accounting for equivalent
        sites.

        A state is characterized by a specific combination of sites. If some
        sites are _equivalent_, then the states are non-unique and, in
        principle, have the same microscopic pKa value. Nonetheless, it may be
        useful to track these separately.

        The number and identity of equivalent states is inferred from:
          1) the number of sites
          2) the number of states
          3) the number of pKa values in the reference compound
          4) the states that are missing compared to all possible states

        Example:
          ASP has two equivalent sites, HD1 and HD2. The site occupancies
          [1, 0] and [0, 1] represent two equivalent states, but can be
          analyzed separately.

        See also:
        ---------
        micro_occupancies_equiv
        """
        occ = self.site_occupancies
        missing_prot_cnts = self.missing_proton_counts
        prot_cnts_are_missing = len(missing_prot_cnts)
        if self.nsites == 1:
            # This is trivial - there can only be two states and one pKa.
            return occ
        elif self.nsites == 2:
            # There are four possible states, but one is often missing.
            if self.nstates == 3 and len(self.pKas) == 2:
                return np.vstack(
                        ((1 - occ[:, 0])*occ[:, 1], occ[:, 0]*(1 - occ[:, 1]))
                ).T
            elif self.nstates == 3 and len(self.pKas) == 1:
                return occ
        elif self.nsites == 3:
            # There are eight possible states, but four are often missing.
            if self.nstates == 4 and prot_cnts_are_missing:
                if all(n in missing_prot_cnts for n in (0, 1)):
                    return np.vstack((
                               (1 - occ[:, 0])*occ[:, 1]*occ[:, 2],
                               occ[:, 0]*(1 - occ[:, 1])*occ[:, 2],
                               occ[:, 0]*occ[:, 1]*(1 - occ[:, 2])
                           )).T
        raise ValueError('site/state combination not implemented')

    @property
    def micro_occupancies_equiv(self):
        """State occupancies for this residue with accounting for equivalent
        sites.

        A state is characterized by a specific combination of sites. If some
        sites are _equivalent_, then the states are non-unique, but can be
        condensed into unique states. These are often, but not always, the same
        as macroscopic states.

        The number and identity of equivalent states is inferred from:
          1) the number of sites
          2) the number of states
          3) the number of pKa values in the reference compound
          4) the states that are missing compared to all possible states

        Example:
          ASP has two equivalent sites, HD1 and HD2. The site occupancies
          [1, 0] and [0, 1] represent two equivalent states. However, a single
          state can be defined as _either_ HD1 or HD2 being occupied.

        See also:
        ---------
        micro_occupancies_noequiv
        """
        occ = self.site_occupancies
        missing_prot_cnts = self.missing_proton_counts
        prot_cnts_are_missing = len(missing_prot_cnts)
        if self.nsites == 1:
            # This is trivial - there can only be two states and one pKa.
            return occ
        elif self.nsites == 2:
            # There are four possible states, but one is often missing.
            if self.nstates == 3 and prot_cnts_are_missing:
                if len(self.pKas) == 1:
                    if 0 in missing_prot_cnts:
                        # Ex. imidazole
                        return occ.prod(axis=1)[:, np.newaxis]
                    elif 2 in missing_prot_cnts:
                        # Ex. carboxylates
                        return occ.sum(axis=1)[:, np.newaxis]
                elif len(self.pKas) == 2:
                    # No equivalent states - Ex. HIS
                    return self.micro_occupancies_noequiv
        elif self.nsites == 3:
            # There are eight possible states, but four are often missing.
            if self.nstates == 4 and prot_cnts_are_missing:
                if all(n in missing_prot_cnts for n in (0, 1)):
                    # Ex. LYS
                    occ = self.micro_occupancies_noequiv
                    return (1 - occ.sum(axis=1))[:, np.newaxis]
        raise ValueError('site/state combination not implemented')

    @property
    def macro_occupancies(self):
        """Macroscopic state occupancies for this residue.

        Macroscopic states combine microscopic states that are
        indistinguishable during a titration. For monoprotic acids with
        equivalent states, the macroscopic and grouped states are the same.
        However, this is not always the case for polyprotic acids.
        """
        occ = self.micro_occupancies_equiv
        if self.nsites == 1:
            return occ
        elif self.nsites == 2:
            if self.nstates == 3:
                if len(self.pKas) == 1:
                    return occ
                else:
                    return (1 - occ.sum(axis=1))[:, np.newaxis]
        elif self.nsites == 3:
            # TODO: this may get buggy when going beyond primary amines...
            if self.nstates == 4:
                return occ
        raise ValueError('site/state combination not implemented')

