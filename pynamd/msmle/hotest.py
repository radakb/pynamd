from __future__ import division

import sys

from numpy import zeros, ones, int32, hstack, newaxis, log, sqrt, arange,\
        all, where
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
        nstates = 12
        ntrials = 1
        
        rmse1a = []
        rmse2a = []
        rmse1e = []
        rmse2e = []
        for n in range(ntrials):
            test = HOSet(nstates, samples_per_state, klow, kmax, mseed, 0.15, 2)
            f3 = test.f_actual

            msmle = MSMLE(test.data, test.data_size)
            msmle.solve_uwham()
            f1 = msmle.f[1:]
            msmle.check_sample_weight_normalization()
            ferr1 = sqrt(msmle.fvar[1:])
            rmse1a.append(sqrt(((f3 - f1)**2).sum()))
            rmse1e.append(sqrt((ferr1**2).mean()))
           
            if do_pymbar:
                try:
                    mbar = MBAR(test.data, test.data_size)
                    f2, ferr2, t = mbar.getFreeEnergyDifferences()
                    f2 = f2[0][1:]
                    ferr2 = ferr2[0][1:]
                    rmse2a.append(sqrt(((f3 - f2)**2).sum()))
                    rmse2e.append(sqrt((ferr2**2).mean()))
                    xbar2, varxbar2 = mbar.computeExpectations(test.x_jn)
                    skipmbar = False
                except:
                    print 'MBAR choked!'
                    skipmbar = True
                    pass
            else:
                skipmbar = True

        if ntrials == 1:
            print 'actual energies',f3

            print 'uwham est. energies',f1
            print 'uwham est. error',ferr1
            print 'uwham actual rms error',rmse1a[0]
            print 'uwham est. rms error',rmse1e[0]

            if not skipmbar:
                print 'mbar est. energies',f2
                print 'mbar est. error',ferr2
                print 'mbar actual rms error',rmse2a[0]
                print 'mbar est. rms error',rmse2e[0]

            print 'actual means'
            print test.x0
            print 'unpooled sample means'
            test_means = zeros(test.nstates)
            test_means[msmle.mask_nonzero] += test.x_jn.mean(axis=1)
            print test_means
            xbar1, varxbar1 = msmle.compute_expectations(test.x_jn)
            print 'pooled sample means'
            print xbar1
            print varxbar1
            if not skipmbar:
                print xbar2
                print varxbar2

        else:
            from numpy import histogram
            H1a, edges = histogram(rmse1a, 20, (0., 5.))
            H2a, edges = histogram(rmse2a, 20, (0., 5.))
            centers = 0.5*(edges[:-1] + edges[1:])
            for x, y1, y2 in zip(centers, H1a, H2a):
                print x, y1, y2

    if dobench:
        for samples_per_state in (100,): #(50, 100, 400, 800):
            for nstates in (1000,): #(80, 160): #(10, 20, 40, 320):
                tu = Timer('test = HOSet(%d, %d, %f, %f); msmle = MSMLE(test.data, test.data_size); msmle.solve_uwham()'%(nstates, samples_per_state, klow, kmax), 'from msmle import MSMLE; from hotest import HOSet')

                tm = Timer('test = HOSet(%d, %d, %f, %f); mbar = MBAR(test.data, test.data_size)'%(nstates, samples_per_state, klow, kmax), 'from pymbar import MBAR; from hotest import HOSet')

                t1 = tu.timeit(1)
                t2 = tm.timeit(1)
                print nstates, samples_per_state, t1, t2, t2/t1

