from __future__ import division
import collections
import json
import warnings

import numpy as np

from pynamd.msmle import MSMLE


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


    _dict = {}
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
    def nresidues(self):
        """The number of residues (same at all pH values)."""
        return self.values()[0].nresidues

    @property
    def residues(self):
        """A list of the residue objects (same at all pH values)."""
        return self.values()[0]

    @property
    def nsites(self):
        """The number of (possibly non-unique) sites (same at all pH values).
        """
        return self.values()[0].nsites

    @property
    def nsamples(self):
        """The number of samples at each pH as an ndarray."""
        return np.asarray([s.nsamples for s in self.itervalues()], np.int32)

    @property
    def total_samples(self):
        """Alias for nsamples.sum()"""
        return self.nsamples.sum()

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

    @property
    def micro_occupancies_noequiv(self):
        """The microstate occupancies (with no adherence to equivalent states)
        from each pH value stacked as a ndarray.

        The shape is (total_samples, nstates) - this is the minimum memory
        shape that can be used for multi-state reweighting methods.
        """
        nstates = self.nstates_micro_noequiv 
        occ = np.zeros((self.total_samples, nstates.sum()), np.int32)
        indices = np.hstack((np.zeros(1, np.int32), self.nsamples.cumsum()))
        for i, j, tsys in zip(indices[:-1], indices[1:], self.itervalues()):
            occ[i:j] += tsys.micro_occupancies_noequiv
        return occ

    @property
    def micro_occupancies_equiv(self):
        """The microstate occupancies (combining equivalent states) from each
        pH value stacked as a ndarray.

        The shape is (total_samples, nstates) - this is the minimum memory
        shape that can be used for multi-state reweighting methods.
        """
        nstates = self.nstates_micro_equiv
        occ = np.zeros((self.total_samples, nstates.sum()), np.int32)
        indices = np.hstack((np.zeros(1, np.int32), self.nsamples.cumsum()))
        for i, j, tsys in zip(indices[:-1], indices[1:], self.itervalues()):
            occ[i:j] += tsys.micro_occupancies_equiv
        return occ

    @property
    def macro_occupancies(self):
        """The macrostate occupancies from each pH value stacked as a ndarray.

        The shape is (total_samples, nstates) - this is the minimum memory
        shape that can be used for multi-state reweighting methods.
        """
        nstates = self.nstates_macro
        occ = np.zeros((self.total_samples, nstates.sum()), np.int32)
        indices = np.hstack((np.zeros(1, np.int32), self.nsamples.cumsum()))
        for i, j, tsys in zip(indices[:-1], indices[1:], self.itervalues()):
            occ[i:j] += tsys.macro_occupancies
        return occ

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

    def compute_titration_curves(self, micro=False, noequiv=False,
                                 est_method='uwham', **kopts):
        """Compute titration curves for the system.

        By default, all residues are treated macroscopically, as if they were
        independently isolated. If micro evaluates True, then the microscopic
        states of each residue are treated separately (the macroscopic states
        are an aggregrate or complement of these states). By default, 
        chemically indistinct (i.e., equivalent) states are not separated 
        (e.g., the protons on carboxylic acids). These states are treated
        separately if noequiv evaluates True. Note that, in many cases, such
        states may not be obviously statistically identical, even within
        reasonable error estimates.

        Arguments
        ---------
        noequiv : bool (default: False)
            If True, analyze non-unique states separately, otherwise aggregate
            those states such that all states are unique.
        est_method : str (default: 'uwham')
            Estimation method for state populations. Supported options are
            'uwham', 'wald', 'yates', and 'agresti-coull'
        kopts 
            Additional keyword options are passed directly to the MSMLE solver
            (see msmle documentation for details). The only exception is the
            'z' keyword, which is interpreted as a confidence interval
            parameter and may affect the extent of bias in non-multi-state
            estimators.

        Returns
        -------
        titration_curves : ndarray
            Two dimensional array of all requested titration curves
        titration_curve_errs : ndarray
            Two dimensional array of standard error estimates for all of the
            requested titration curves
        """
        if micro:
            if noequiv:
                occs = self.micro_occupancies_noequiv
            else:
                occs = self.micro_occupancies_equiv
        else:
            occs = self.macro_occupancies
        nstates = occs.shape[1]
        titration_curves = np.zeros((nstates, self.numpHs))
        titration_curve_errs = np.zeros((nstates, self.numpHs))
        # The confidence parameter, z, is not a standard argument for MSMLE,
        # but is the only parameter for other estimation types. Nonetheless, it
        # can be used after the fact as a scale factor for the error.
        #
        try:
            z2 = float(kopts.pop('z'))**2
        except KeyError:
            z2 = 1.0

        _method = str(est_method).lower()
        if _method in self._MULTISTATE_METHODS:
            msmle_ = self._compute_multistate_weights(_method, **kopts)
            warnings.simplefilter("error", RuntimeWarning)
            # Iterate each state and compute the titration curve at all pHs.
            for i, occ in enumerate(occs.T):
                p, pvar = msmle_.compute_expectations(occ, True)
                titration_curves[i] += p
                try:
                    titration_curve_errs[i] += np.sqrt(z2*pvar)
                except RuntimeWarning:
                    titration_curve_errs[i] = np.nan
        else:
            if _method in ('wald', 'naive', 'yates'):
                def get_pop(occ):
                    # The Yates method essentially adds a correction to the
                    # standard Wald estimate and then re-scales it. It is also
                    # sometimes called the Wilson score interval or just the
                    # Wilson interval.
                    n, p = occ.shape[0], occ.mean(axis=0)
                    if _method == 'yates':
                        p = (2*n*p + z2) / (2*(n + z2))
                    q = 1 - p
                    pvar = z2*p*q / n
                    return p, pvar
            elif _method == 'agresti-coull':
                def get_pop(occ):
                    # The Agresti-Coull method essentially redefines the sample
                    # size and shifts the data.
                    n = occ.shape[0] + z2
                    p = (occ.sum(axis=0) + 0.5*z2) / n
                    q = 1 - p
                    pvar = z2*p*q / n
                    return p, pvar
            else:
                raise ValueError('Unrecognized method %s'%str(est_method))
            # Iterate each pH and compute the titration curve for each state. 
            indices = np.hstack((np.zeros(1, np.int32),
                                 self.nsamples.cumsum()
                               ))
            for j, (n1, n2) in enumerate(zip(indices[:-1], indices[1:])):
                p, pvar = get_pop(occs[n1:n2])
                titration_curves[:, j] += p
                titration_curve_errs[:, j] += np.sqrt(pvar)
        return titration_curves, titration_curve_errs

    def _compute_multistate_weights(self, est_method, **kopts):
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
            msmle.solve_uwham(**kopts)
        return msmle


class TitratableSystem(list):
    """A system of multiple titratable residues in contact with the same pH

    Parameters
    ----------
    pH : float
        The pH value that the residues are in contact with
    titratable_residues : TitratableResidue
        One or more TitratableResidue objects
    """
    def __init__(self, pH, *titratable_residues):
        self.pH = _validate_float(pH)
        list.__init__(self, *titratable_residues)

    def append(self, titratable_residue):
        if not isinstance(titratable_residue, TitratableResidue):
            raise TypeError('Can only add TitratableResidue objects.')
        list.append(self, titratable_residue)

    def subsample(self, start, stop, step):
        """Modify the data set for all residues in-place."""
        for tres in self:
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
        return np.hstack((tres.site_occupancies for tres in self)).sum(axis=1)

    @property
    def nsites(self):
        """The number sites spanned by per residue"""
        return np.asarray([tres.nsites for tres in self])

    @property
    def nsamples(self):
        """The number of occupation vector samples"""
        # TODO: Check that this is actually true for all residues?
        #       This should never not be the case.
        return self[0].nsamples

    @property
    def nstates_micro_noequiv(self):
        """The number of microstates per residue without equivalencing"""
        return np.asarray([tres.nstates_micro_noequiv for tres in self])

    @property
    def nstates_micro_equiv(self):
        """The number of microstates per all residue with equivalencing"""
        return np.asarray([tres.nstates_micro_equiv for tres in self])

    @property
    def nstates_macro(self):
        """The number of macrostates per residues"""
        return np.asarray([tres.nstates_macro for tres in self])

    @property
    def micro_occupancies_noequiv(self):
        nstates = self.nstates_micro_noequiv
        occ = np.zeros((self.nsamples, nstates.sum()), np.int32)
        i = 0
        for tres, n in zip(self, nstates):
            j = i + n
            occ[:, i:j] += tres.micro_occupancies_noequiv
            i = j
        return occ

    @property
    def micro_occupancies_equiv(self):
        nstates = self.nstates_micro_equiv
        occ = np.zeros((self.nsamples, nstates.sum()), np.int32)
        i = 0
        for tres, n in zip(self, nstates):
            j = i + n
            occ[:, i:j] += tres.micro_occupancies_equiv
            i = j
        return occ

    @property
    def macro_occupancies(self):
        nstates = self.nstates_macro
        occ = np.zeros((self.nsamples, nstates.sum()), np.int32)
        i = 0
        for tres, n in zip(self, nstates):
            j = i + n
            occ[:, i:j] += tres.macro_occupancies
            i = j
        return occ
 
    def __repr__(self):
        return 'TitratableSystem(%s, %s)'%(str(self.pH), list.__repr__(self))

    def __eq__(self, other):
        if (self.pH != other.pH) or (len(self) != len(other)):
            return False
        for (sres, ores) in zip(self, other):
            if sres.segresidname != ores.segresidname:
                return False
        return True

    def __iadd__(self, other):
        """In-place addition - concatenate occupancies if otherwise equal."""
        if not self == other:
            raise TypeError('Cannot add - mismatch in system pH or residues.')
        for (sres, ores) in zip(self, other):
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
        pH, res_strs, occ = TitratableSystem._read_cphlog(cphlog)
        obj = cls(pH)
        i = 0
        for segresidname in res_strs:
            segid, resid, resname = segresidname.split(':')
            states = json_data[resname]['states']
            pKas = json_data[resname]['pKa']
            nsites = len(states.itervalues().next())
            j = i + nsites
            obj.append(
                TitratableResidue(segresidname, states, pKas, occ[:, i:j])
            )
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
        self.segresidname = str(segresidname)
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
        return ('TitratableResidue(%s, %s, %s)'
                %(self.segresidname, str(self.states), str(self.pKas))
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
            if self.nstates == 3:
                return np.vstack(
                        ((1 - occ[:, 0])*occ[:, 1], occ[:, 0]*(1 - occ[:, 1]))
                ).T
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

