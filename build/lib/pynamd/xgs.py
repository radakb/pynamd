"""Derived class for NAMD XGS log file output"""
from numpy import asarray, concatenate, int32, any

from pynamd.log import NamdLog


class NamdXGSLog(NamdLog):
    def __init__(self, *filenames, **kwargs):
        _filenames = [str(filename) for filename in filenames]
        basefile = _filenames[0]
        NamdLog.__init__(self, basefile, **kwargs)
        self.xgs = {}
        xgstype, ladder, weights, state = NamdXGSLog.read_xgs(basefile)
        self.xgs['type'] = xgstype
        self.xgs['ladder'] = ladder
        self.xgs['weights'] = weights
        self.xgs['state'] = state
        # Add additional files by recursive in-place addition. 
        self.filenames = _filenames
        for filename in self.filenames[1:]:
            self += NamdXGSLog(filename, **kwargs)

    @property
    def ncycles(self):
        return self.xgs['state'].size

    @property
    def efreq(self):
        return self.energy['TS'][1:].size // self.ncycles

    @staticmethod
    def read_xgs(filename):
        """Read XGS information from a trajectory.

        WARNING! EXPERIMENTAL! NOT GUARANTEED TO WORK AS EXPECTED!
                 THE XGS OUTPUT FORMAT IS NOT FIXED YET!
        """
        TAG = 'TCL: XGS)'
        xgstype, ladder, weights, indices = None, None, None, []
        for line in open(filename, 'r'):
            if not line.startswith(TAG):
                continue
            _line = line.lstrip(TAG).strip()
            if _line.startswith('simulation type:'):
                xgstype = ' '.join(_line.split()[2:]).upper()
            elif _line.startswith('state parameters:'):
                terms = _line.split()[2:]
                try:
                    ladder = [float(p) for p in terms]
                except ValueError:
                    _line = ' '.join(terms).lstrip('{').rstrip('}')
                    terms = _line.split('} {')
                    ladder = [[float(p) for p in s.split()] for s in terms]
                ladder = asarray(ladder)
            elif _line.startswith('state weights:'):
                weights = asarray([float(w) for w in _line.split()[2:]])
            elif _line.startswith('cycle'):
                terms = _line.split()
                indices.append(int(terms[-1]))
        if xgstype not in ('ST', 'ALCH', 'SS', 'ALCHST', 'PDS'):
            raise ValueError('Bad XGS simulation type %s'%xgstype)
        if ladder.shape[0] != weights.size:
            raise ValueError('Mismatch between ladder and weight sizes')
        indices = asarray(indices, int32)
        return (xgstype, ladder, weights, indices)

    def __eq__(self, other):
        if not NamdLog.__eq__(self, other):
            return False
        for key in ('type', 'ladder', 'weights'):
            if any(self.xgs[key] != other.xgs[key]):
                return False
        return True

    def __iadd__(self, other):
        if not self == other:
            raise TypeError('Mismatch in terms. NamdXGSLogs are incompatible!')
        NamdLog.__iadd__(self, other)
        key = 'state'
        self.xgs[key] = concatenate((self.xgs[key], other.xgs[key]))
        return self

