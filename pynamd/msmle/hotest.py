from __future__ import division, print_function

import sys

from numpy import zeros, ones, int32, hstack, newaxis, log, sqrt, arange,\
        all, abs, where
from numpy.random import uniform, normal, seed, randint, choice, exponential


class HOSet(object):
    """
    nstates - number of HO states to sample
    max_samples - maximum number of samples to draw from a given HO
    klow - minimum force constant
    khi - maximum force constant
    randeed - number generator seed (for repeatable systems)
    sample_fudge - percent fudge in reduction from max_samples
    """
    def __init__(self, nstates=2, max_samples=10, klow=1.0e-1, khi=1.0e1,
                 randseed=None, sample_fudge=0.0, unsampled_states=0):
        self.max_samples = int(max_samples)
        self.nstates = int(nstates)
        # Randomize the HO parameters.
        seed(randseed)
        klow, khi = float(klow), float(khi)
        #spacing = uniform(self.nstates, size=self.nstates)
        #k = klow*(khi / klow)**(spacing / self.nstates)
        # k = uniform(float(klow), float(khi), size=self.nstates)
        k = klow + (khi - klow)*exponential(1.0, self.nstates)
        sigma = sqrt(1/k)
        x0 = uniform(-0.5*sigma.max(), 0.5*sigma.max(), size=self.nstates)
        # Choose which states to sample from.
        nsampled_states = self.nstates - int(unsampled_states)
        sampled_indices = choice(arange(self.nstates), nsampled_states, False)
        sampled_indices.sort()
        # Generate samples up to max.
        x_in = normal(0.0, 1.0, (nsampled_states, self.max_samples))
        x_in *= sigma[sampled_indices, newaxis]
        x_in += x0[sampled_indices, newaxis]
        self.data_size = zeros(self.nstates, int32) 
        self.data_size[sampled_indices] += self.max_samples
        # Randomly remove samples for uneven sampling.  Note that at least one
        # state must remain the same, otherwise max_samples is incorrect.
        # Also, we don't actually have to do anything to the actual samples, bc
        # the sample size is used as a mask!
        #
        del_max = int(sample_fudge*self.max_samples + 0.5) + 1
        if del_max > 1:
            sample_shift = randint(0, del_max, nsampled_states)
            if all(sample_shift > 0): # Randomly reset the shift for a state.
                sample_shift[choice(arange(nsampled_states))] = 0
            self.data_size[sampled_indices] -= sample_shift
        self.unsampled_indices = where(self.data_size == 0)[0]
        # Compute the energy in all states
        u_ijn = 0.5*(k[:, newaxis]*(x_in[:, newaxis, :] - x0[:, newaxis])**2)
        self.u_ijn = u_ijn
        self.f_actual = 0.5*log(k / k[0])[1:]
        self.x0 = x0
        self.x_jn = x_in

    @property
    def data(self):
        return self.u_ijn

    @property
    def unsampled_data(self):
        """Return just the energies from states with no samples"""
        idx = self.unsampled_indices
        u_jn_k = zeros((idx.size, self.data_size.sum()))
        shift = 0
        for i, n in enumerate(self.data_size):
            if n == 0:
                shift += 1
                continue
            j = self.data_size[:i].sum()
            u_jn_k[:, j:(j+n)] = self.u_ijn[i-shift, idx, :n]
        return u_jn_k 

if __name__ == '__main__':
    import sys
    import copy
    from timeit import Timer

    from msmle import MSMLE
    try:
        from pymbar import MBAR
        do_pymbar = True
    except ImportError:
        do_pymbar = False

    try:
        mseed = int(sys.argv[1])
    except IndexError:
        mseed = None

    klow = 1.0e-3
    kmax = 5.0e2
    dotest = True
    dobench = False

    if dotest:
        samples_per_state = 500
        nstates = 8
        bootstrap_trials = 200

        test = HOSet(nstates, samples_per_state, klow, kmax, mseed, 0.15, 2)
        f3 = test.f_actual

        msmle = MSMLE(test.data, test.data_size)
        msmle.solve_uwham()
        f1 = msmle.f
        f1 -= f1[0]
        ferr1 = sqrt(msmle.fvar[1:])    
        xbar1, varxbar1 = msmle.compute_expectations(test.x_jn)

        ferr1_bs = zeros(f1.size)
        varxbar1_bs = zeros(xbar1.size)
        if bootstrap_trials > 1:
            f1_bs = zeros((bootstrap_trials, f1.size))
            xbar1_bs = zeros((bootstrap_trials, xbar1.size))
            for trial in xrange(bootstrap_trials):
                msmle.resample()
                msmle.solve_uwham(f1)
                f1_bs[trial] = msmle.f
                f1_bs[trial] -= msmle.f[0]
                xbar1_bs[trial] = msmle.compute_expectations(test.x_jn, False)[0]
            ferr1_bs = f1_bs.std(axis=0)[1:]
            varxbar1_bs = xbar1_bs.var(axis=0)
            msmle.revert_sample()
        f1 = f1[1:]

        if do_pymbar:
            try:
                mbar = MBAR(test.data, test.data_size)
                f2, ferr2, t = mbar.getFreeEnergyDifferences()
                f2 = f2[0][1:]
                ferr2 = ferr2[0][1:]
                xbar2, varxbar2 = mbar.computeExpectations(test.x_jn)
                skipmbar = False
            except:
                print('MBAR choked!')
                skipmbar = True
                pass
        else:
            skipmbar = True

        def print_float_array(msg, arr):
            print('%-16s '%msg + ' '.join(('% 6.4f'%x for x in arr)))

        print('samples:', test.data_size)
        print_float_array('actual energies', f3)

        print_float_array('uwham energies', f1)
        print_float_array('uwham bst mean', f1_bs.mean(axis=0)[1:])
        print_float_array('uwham act. err', abs(f1 - f3))
        print_float_array('uwham est. err', ferr1)
        print_float_array('uwham bst. err', ferr1_bs)

        if not skipmbar:
            print_float_array('mbar energies', f2)
            print_float_array('mbar est. err', ferr2)

        print_float_array('actual means', test.x0)
        test_means = zeros(test.nstates)
        test_means[msmle.mask_nonzero] += test.x_jn.mean(axis=1)
        print_float_array('unweighted means', test_means)
        print_float_array('uwham means', xbar1)
        print_float_array('act. err', abs(xbar1 - test.x0))
        print_float_array('est. err', sqrt(varxbar1))
        print_float_array('bst. err', sqrt(varxbar1_bs))
        if not skipmbar:
            print_float_array('mbar means', xbar2)
            print_float_array('act. err', abs(xbar2 - test.x0))
            print_float_array('est. ett', sqrt(varxbar2))

    if dobench:
        for samples_per_state in (100,): #(50, 100, 400, 800):
            for nstates in (1000,): #(80, 160): #(10, 20, 40, 320):
                tu = Timer('test = HOSet(%d, %d, %f, %f); msmle = MSMLE(test.data, test.data_size); msmle.solve_uwham()'%(nstates, samples_per_state, klow, kmax), 'from msmle import MSMLE; from hotest import HOSet')

                tm = Timer('test = HOSet(%d, %d, %f, %f); mbar = MBAR(test.data, test.data_size)'%(nstates, samples_per_state, klow, kmax), 'from pymbar import MBAR; from hotest import HOSet')

                t1 = tu.timeit(1)
                t2 = tm.timeit(1)
                print(nstates, samples_per_state, t1, t2, t2/t1)

