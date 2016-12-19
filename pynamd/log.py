"""Basic class for interacting with NAMD standard output as a log file"""
from __future__ import division
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from requests.structures import CaseInsensitiveDict
from math import sqrt

from numpy import asarray, ones, concatenate, int32, minimum, maximum


class NamdLog(object):
    """The NamdLog class provides a simple way to interact with output from 
    NAMD that is sent to stdout (usually redirected into a so-called log file).

    The two main attributes are an "info" dict that stores the keyword/value
    settings as they would be found in the configuration file (in principle
    this can be used to recapitulate the input) and an "energy" dict that
    stores the energy time series information as numpy arrays. Advanced
    simulation methods may also produce additional dict attributes.

    By virtue of the abstraction, NamdLog objects can only be initialized from
    NAMD log files. Multiple NamdLog objects can be glommed together as a
    single object during construction or by in-place addition. Obviously, order
    matters. Consistency checking between files only exists in so far as the
    same kinds of output are present - simulation settings are not examined at
    all.

    For most use cases, the main attribute of interest is the 'energy' dict.
    This provides direct access to the time series of each 'ENERGY:' field
    using the standard NAMD keywords. These are the raw output, as-is, and do
    not check for things like redundant time steps. The only time anything
    remotely clever is done is when two objects are added in-place, in which
    case it is assumed that the last step of the first object is redundant with
    step 0 of the second. This is probably the most commonly desired behavior,
    but certainly not completely general.

    The other most useful attribute is the 'info' dict, although this is much
    less robust. The keywords for 'info' are the same as those in NAMD and are
    also case-insensitive (only a common subset of keywords are actually
    available, but these can be added relatively easily by request).

    Generic querying of certain parameters is also available. For example, the
    'temperature' attribute points to the value of _any_ thermostat temperature
    that has been set. This can also be queried as a NAMD consistent value of
    kT, without regard for how temperature was actually maintained. Of course,
    NVE simulations have no formal temperature and thus these return None in
    that case.

    Example - three files: out0.log, out1.log, and out2.log

    >>> mylog = NamdLog('out0.log', 'out1.log', 'out2.log')

    OR
 
    >>> mylog = NamdLog('out0.log')
    >>> for logfile in ('out1.log', 'out2.log'):
    >>>     mylog += NamdLog(logfile)
    
    >>> mylog.energy['TOTAL'] # returns an array of the total energy values
    array([-16171.4344, -16192.6294, -16118.3861, ..., -16344.6238,
       -16290.4998, -16343.054 ])
    >>> mylog.energy['ELECT'] # returns an array of the electrostatic energies
    array([-21810.3769, -21934.0999, -21733.8116, ..., -22035.4452,
       -21942.0233, -21877.0512])
    >>> mylog.info['langevin']
    True
    >>> mylog.info['Langevin']
    True
    >>> mylog.info['cutoff']
    12.0
    >>> mylog.temperature
    298.0
    >>> # compute 'thermodynamic beta' for this ensemble
    >>> 1 / mylog.kT
    1.688667419481357
    >>> # This also lists the available keywords for mylog.energy.
    >>> mylog.etitle 
    'ETITLE:      TS           BOND          ANGLE          DIHED          IMPRP               ELECT            VDW       BOUNDARY           MISC        KINETIC               TOTAL           TEMP      POTENTIAL         TOTAL3        TEMPAVG            PRESSURE      GPRESSURE         VOLUME       PRESSAVG      GPRESSAVG'
    >>> # Entries are the average of steps 1-100, using every second step.
    >>> mylog.energy_mean(1, 100, 2)
    'ENERGY:       0        10.5427        13.5865         6.3911         1.2225         -21664.4380      2072.4712         0.0000         0.0000      3540.7685         -16019.4555       297.6854    -19560.2240    -16012.1968       297.3508            -86.0568       -85.2919     58895.1677       -98.5552       -98.5654'
    """
    BOLTZMANN = 0.001987191 # from NAMD common.h, in kcal/mol-K

    # It is usually not possible to determine at read time that a parameter 
    # should be stored as a list of multiple values. This list tells us which
    # parameters fit this description ahead of time.
    #
    _multivals = ['parameters', 'topology']

    # Special typing functions for unusual settings. 
    #
    def _exclude2str(setting):
        """Translate the weird exclude settings."""
        options = {'NONE': 'none', 'ONETWO': '1-2', 'ONETHREE': '1-3',
                   'SCALED': 'scaled1-4'}
        try:
            return options[setting]
        except KeyError:
            raise Exception("Unknown 'exclude' type '%s'."%setting)

    def _alchtype(atype):
        """Alchemical settings have a bool flag _and_ a type; set both."""
        return ['on', atype.lstrip('(').rstrip(')').lower()]

    def _vec2list(dtype):
        """Return a function that converts a list of strings to a list of the
        desired type. If a value cannot be converted, the function returns None
        in its place (thus the input and output lists have corresponding
        elements).
        """
        def typedvec(vec):
            tvec = []
            for v in vec:
                try:
                    tvec.append(dtype(v))
                except (ValueError, TypeError):
                    tvec.append(None)
            return tvec
        return typedvec

    """The _kvpairs dict is meant to allow easy addition of keyword/value pairs
    to be looked up in a log file. Whenever a line prefaced by 'Info:' is 
    encountered, check against all of the keys in _kvpairs until a match is
    found. The value is then stored according to the corresponding keyword
    value from the config file. 

    Lookup is done using the string after "Info:" in the NAMD log. The value is
    a three element tuple of: 1) the accompanying configuration file keyword
    (this cannot, in general, be inferred from the output), 2) the number of
    values to be stored (-1 for a bool*, 1 for a scalar, and more for a
    vector), and 2) the type of the value(s) (e.g. int, float). 

    *In the case of a bool there usually is no value, in which case _this_ is
    the value to be set (i.e. True or False).

    In my opinion this is _much_ nicer than having a crazy if/else statement in
    the read_info() method. Minor changes could also expose _kvpairs as an
    attribute to the user and allow modifications on the fly, but if that were
    really desired you are probably competent/curious enough to be reading
    this and making the change here permanently.

    TODO: If _kvpairs ever gets so large that it becomes noticeably slow to
    keep iterating through it, then keys could be deleted as they are found
    (assuming only one value is possible).
    """
    _kvpairs = (
        #
        # Force fields and PSF/PRM vs PRMTOP
        #
        # CHARMM/XPLOR input files
        ('PARAMETER file: CHARMM format!', 'paraTypeCharmm', -1, True),
        ('PARAMETER file: XPLOR format!', 'paraTypeXplor', -1, True),
        ('PARAMETERS', 'parameters', 1, str),
        ('STRUCTURE FILE', 'structure', 1, str),
        ('COORDINATE PDB', 'coordinates', 1, str),
        ('BINARY COORDINATES', 'binCoordinates', 1, str),
        ('VELOCITY FILE', 'binVelocities', 1, str),
        ('EXTENDED SYSTEM FILE', 'extendedSystem', 1, str),
        ('INITIAL TEMPERATURE', 'temperature', 1, float),
        # AMBER input files
        ('Using AMBER format force field!', 'amber', -1, True),
        ('AMBER PARM FILE', 'parmfile', 1, str),
        ('AMBER COORDINATE FILE', 'ambercoor', 1, str),
        ('Exclusions will be read from PARM file!', 'readexclusions', -1, 
         True),
        ('Exclusions in PARM file will be ignored!', 'readexclusions', -1,
         False),
        ('SCNB (VDW SCALING)', 'scnb', 1, float),
        #
        # Drude
        #
        ('DRUDE MODEL DUAL THERMOSTAT IS ACTIVE', 'drude', -1, True),
        ('DRUDE BOND TEMPERATURE', 'drudeTemp', 1, float),
        ('DRUDE DAMPING COEFFICIENT IS', 'drudeDamping', 1, float),
        ('DRUDE MAXIMUM BOND LENGTH BEFORE RESTRAINT IS', 'drudeBondLen', 1,
         float),
        ('DRUDE BOND RESTRAINT CONSTANT IS', 'drudeBondConst', 1, float),
        ('DRUDE HARD WALL RESTRAINT IS ACTIVE FOR DRUDE BONDS', 
         'drudeHardWall', -1, True),
        ('DRUDE NBTHOLE RADIUS IS', 'drudeNBTholeCut', 1, float),
        #
        # Boundary conditions
        #
        ('PERIODIC CELL BASIS 1', 'cellBasisVector1', 3, _vec2list(float)),
        ('PERIODIC CELL BASIS 2', 'cellBasisVector2', 3, _vec2list(float)),
        ('PERIODIC CELL BASIS 3', 'cellBasisVector3', 3, _vec2list(float)),
        #
        # Output formatting
        #
        ('OUTPUT FILENAME', 'outputname', 1, str),
        ('WRAPPING WATERS AROUND PERIODIC BOUNDARIES ON OUTPUT.', 'wrapWater',
         -1, True),
        ('WRAPPING ALL CLUSTERS AROUND PERIODIC BOUNDARIES ON OUTPUT.',
         'wrapAll', -1, True),
        ('WRAPPING TO IMAGE NEAREST TO PERIODIC CELL CENTER.', 'wrapNearest',
         -1, True),
        ('ENERGY OUTPUT STEPS', 'outputEnergies', 1, int),
        ('DCD FREQUENCY', 'DCDFreq', 1, int),
        ('TIMING OUTPUT STEPS', 'outputTiming', 1, int),
        ('FIRST TIMESTEP', 'firstTimestep', 1, int),
        #
        # Integrators, thermostats, and constraints
        #
        ('STEPS PER CYCLE', 'stepsPerCycle', 1, int),
        ('TIMESTEP', 'timestep', 1.0, float),
        ('NUMBER OF STEPS', 'numsteps', 1, int),
        ('RIGID BONDS TO HYDROGEN :', 'rigidBonds', 1, str),
        ('ERROR TOLERANCE :', 'rigidTolerance', 1, float),
        ('MAX ITERATIONS :', 'rigidIterations', 1, int),
        ('RIGID WATER USING SETTLE ALGORITHM', 'useSettle', -1, True),
        ('NONBONDED FORCES EVALUATED EVERY', 'nonbondedFreq', 1, int),
        ('FULL ELECTROSTATIC EVALUATION FREQUENCY', 'fullElectFrequency', 1,
         int),
        ('RANDOM NUMBER SEED', 'seed', 1, int),
        # Langevin dynamics
        ('LANGEVIN DYNAMICS ACTIVE', 'langevin', -1, True),
        ('LANGEVIN TEMPERATURE', 'langevinTemp', 1, float),
        ('LANGEVIN DAMPING COEFFICIENT IS', 'langevinDamping', 1, float),
        ('LANGEVIN DYNAMICS APPLIED TO HYDROGENS', 'langevinHydrogen', -1, 
         True),
        ('LANGEVIN DYNAMICS NOT APPLIED TO HYDROGENS', 'langevinHydrogen', -1,
         False),
        # Lowe-Andersen thermostat
        ('LOWE-ANDERSEN DYNAMICS ACTIVE', 'loweAndersen', -1, True),
        ('LOWE-ANDERSEN TEMPERATURE', 'loweAndersenTemp', 1, float),
        ('LOWE-ANDERSEN RATE', 'loweAndersenRate', 1, float),
        ('LOWE-ANDERSEN CUTOFF', 'loweAndersenCutoff', 1, float),
        # temperature coupling
        ('TEMPERATURE COUPLING ACTIVE', 'tCouple', -1, True),
        ('COUPLING TEMPERATURE', 'tCoupleTemp', 1, float),
        # velocity rescaling
        ('VELOCITY RESCALE FREQ', 'rescaleFreq', 1, int),
        ('VELOCITY RESCALE TEMP', 'rescaleTemp', 1, float),
        # velocity reassignment
        ('VELOCITY REASSIGNMENT FREQ', 'reassignFreq', 1, int),
        ('VELOCITY REASSIGNMENT TEMP', 'reassignTemp', 1, float),
        #
        # barostats
        #
        ('PRESSURE CONTROL IS GROUP-BASE', 'useGroupPressure', -1, True),
        ('PRESSURE CONTROL IS ATOM-BASE', 'useGroupPressure', -1, False),
        ('CELL FLUCTUATION IS ANISOTROPIC', 'useFlexibleCell', -1, True),
        ('CELL FLUCTUATION IS ISOTROPIC', 'useFlexibleCell', -1, False),
        ('SHAPE OF CELL IS CONSTRAINED IN X-Y PLANE', 'useConstantRatio', -1,
         True),
        ('CONSTANT AREA PRESSURE CONTROL ACTIVE', 'useConstantArea', -1, True),
        ('TARGET SURFACE TENSION IS', 'surfaceTensionTarget', 1, float),
        # Langevin piston
        ('LANGEVIN PISTON PRESSURE CONTROL ACTIVE', 'langevinPiston', -1, 
         True),
        ('TARGET PRESSURE IS', 'langevinPistonTarget', 1, float),
        ('OSCILLATION PERIOD IS', 'langevinPistonPeriod', 1, float),
        ('DECAY TIME IS', 'langevinPistonDecay', 1, float),
        ('PISTON TEMPERATURE IS', 'langevinPistonTemp', 1, float),
        #
        # Electrostatics
        #
        ('SWITCHING ACTIVE', 'switching', -1, True),
        ('VDW FORCE SWITCHING ACTIVE', 'vdwForceSwitching', -1, True),
        ('LONG-RANGE LJ:', 'LJcorrection', -1, True),
        ('SWITCHING ON', 'switchdist', 1, float),
        ('SWITCHING OFF', 'cutoff', 1, float),
        ('CUTOFF', 'cutoff', 1, float),
        ('PAIRLIST DISTANCE', 'pairlistdist', 1, float),
        ('PAIRLISTS PER CYCLE', 'pairlistsPerCycle', 1, int),
        ('EXCLUDE', 'exclude', 1, _exclude2str),
        ('1-4 ELECTROSTATICS SCALED BY', '1-4scaling', 1, float),
        # PME
        ('PARTICLE MESH EWALD (PME) ACTIVE', 'PME', -1, True),
        ('PME TOLERANCE', 'PMETolerance', 1, float),
        ('PME INTERPOLATION ORDER', 'PMEInterpOrder', 1, int),
        ('PME GRID DIMENSIONS', ('PMEGridSizeX', 'PMEGridSizeY', 
         'PMEGridSizeZ'), 3, _vec2list(int)),
        ('PME MAXIMUM GRID SPACING', 'PMEGridSpacing', 1, float),
        # GB
        ('GBIS GENERALIZED BORN IMPLICIT SOLVENT ACTIVE', 'GBIS', -1, True),
        ('GBIS BORN RADIUS CUTOFF:', 'alphaCutoff', 1, float),
        ('GBIS ION CONCENTRATION:', 'ionConcentration', 1, float),
        #
        # Alchemy (Note that these have parallel/redundant output)
        #
        # FEP
        ('ALCHEMICAL', ('alch', 'alchType'), 1, _alchtype),
        ('FEP CURRENT LAMBDA VALUE', 'alchLambda', 1, float),
        ('FEP COMPARISON LAMBDA VALUE', 'alchLambda2', 1, float),
        ('FEP CURRENT LAMBDA VALUE SET TO INCREASE IN EVERY', 'alchLambdaFreq',
         1, float),
        ('FEP INTRA-ALCHEMICAL NON-BONDED INTERACTIONS WILL BE DECOUPLED',
         'alchDecouple', -1, False),
        ('FEP INTRA-ALCHEMICAL NON-BONDED INTERACTIONS WILL BE RETAINED',
         'alchDecouple', -1, True),
        ('FEP INTRA-ALCHEMICAL BONDED INTERACTIONS WILL BE DECOUPLED',
         'alchBondDecouple', -1, True),
        ('FEP INTRA-ALCHEMICAL BONDED INTERACTIONS WILL BE RETAINED',
         'alchBondDecouple', -1, False),
        ('FEP VDW SHIFTING COEFFICIENT', 'alchVdwShiftCoeff', 1, float),
        ('FEP ELEC. ACTIVE FOR EXNIHILATED PARTICLES BETWEEN LAMBDA =',
         'alchElecLambdaStart', 1, float),
        (('FEP VDW ACTIVE FOR EXNIHILATED PARTICLES BETWEEN LAMBDA = 0 AND '
          'LAMBDA ='), 'alchVdwLambdaEnd', 1, float),
        (('FEP BOND ACTIVE FOR EXNIHILATED PARTICLES BETWEEN LAMBDA = 0 AND '
          'LAMBDA ='), 'alchBondLambdaEnd', 1, float),
        # TI
        ('THERMODYNAMIC INTEGRATION', ('alch', 'alchType'), 1, _alchtype),
        ('TI LAMBDA VALUE', 'alchLambda', 1, float),
        ('TI COMPARISON LAMBDA VALUE', 'alchLambda2', 1, float),
        ('TI CURRENT LAMBDA VALUE SET TO INCREASE IN EVERY', 'alchLambdaFreq',
         1, float),
        ('TI INTRA-ALCHEMICAL NON-BONDED INTERACTIONS WILL BE DECOUPLED',
         'alchDecouple', -1, False),
        ('TI INTRA-ALCHEMICAL NON-BONDED INTERACTIONS WILL BE RETAINED',
         'alchDecouple', -1, True),
        ('TI INTRA-ALCHEMICAL BONDED INTERACTIONS WILL BE DECOUPLED',
         'alchBondDecouple', -1, True),
        ('TI INTRA-ALCHEMICAL BONDED INTERACTIONS WILL BE RETAINED',
         'alchBondDecouple', -1, False),
        ('TI VDW SHIFTING COEFFICIENT', 'alchVdwShiftCoeff', 1, float),
        ('TI ELEC. ACTIVE FOR ANNIHILATED PARTICLES BETWEEN LAMBDA =',
         'alchElecLambdaStart', 1, float),
        (('TI VDW ACTIVE FOR EXNIHILATED PARTICLES BETWEEN LAMBDA = 0 AND '
          'LAMBDA ='), 'alchVdwLambdaEnd', 1, float),
        (('TI BOND ACTIVE FOR EXNIHILATED PARTICLES BETWEEN LAMBDA = 0 AND '
          'LAMBDA ='), 'alchBondLambdaEnd', 1, float),
        #
        # Enhanced Sampling
        #
        ('ACCELERATED MD ACTIVE', 'accelMD', -1, True),
        ('BOOSTING DIHEDRAL POTENTIAL', 'accelMDDihe', -1, True),
        ('accelMDE:', ('accelMDE', None, None, 'accelMDAlpha'), 4,
         _vec2list(float)),
        ('accelMD WILL BE DONE FROM STEP', 'accelMDFirstStep', 1, int),
        ('accelMD OUTPUT FREQUENCY', 'accelMDOutFreq', 1, int)
    )

    # Any given NamdLog may or may not have the following attributes
    # depending on the simulation settings.
    _attr_names = ['energy', 'ti']
#    _attr_names = ['energy', 'amd_energy', 'ti']

    def __init__(self, *filenames, **kwargs):
        #     Assign default values to optional keyword arguments. This can be
        # done more elegantly at the expense of Python 2.x support.
        #
        info = (bool(kwargs['info']) if 'info' in kwargs else True)
        energy = (bool(kwargs['energy']) if 'energy' in kwargs else True) 
        xgs = (bool(kwargs['xgs']) if 'xgs' in kwargs else False)
        self._nostep0 = False 
        self.filenames = [str(filename) for filename in filenames]
        basefile = self.filenames[0]
        # Read INFO: lines from the header or not.
        if info:
            self.info = NamdLog.read_info(basefile)
        else: 
            self.info = None
        # Read ENERGY: lines or not.
        if energy:
            energies = NamdLog.read_energy(basefile)
            for kw, value in energies.iteritems():
                self.__dict__[kw] = value
        else:
            self.energy = None
        # Read XGS specific output - EXPERIMENTAL!
        if xgs:
            self.xgs = {}
            xgstype, ladder, weights, state = NamdLog.read_xgs(basefile)
            self.xgs['type'] = xgstype
            self.xgs['ladder'] = ladder
            self.xgs['weights'] = weights
            self.xgs['state'] = state
        # Add additional files by recursive in-place addition. 
        for filename in self.filenames[1:]:
            self += NamdLog(filename, **kwargs)

    def __repr__(self):
        args = ', '.join(["'%s'"%f for f in self.filenames])
        return '%s(%s)'%(self.__class__, args) 
 
    @staticmethod
    def read_info(filename):
        """Return a dict of keyword/value pairs after parsing the 'Info:' 
        lines of a NAMD output log. Dict queries are case-insensitive.
        """
        info = {}

        def add_info(key, value):
            """Helper function for handling list values versus other types."""
            if key in NamdLog._multivals: # info[key] is a list of values
                try:
                    info[key].append(value)
                except KeyError:
                    info[key] = info.get(key, [value])
            else: # info[key] is a single value
                info[key] = value

        def parse(tokens, key, configkey, nvalfields, convert):
            """Helper function for handling different keyword/value formats
            NAMD does not have a strict standard on this output format.
            """
            nkeyfields = len(key.split())
            test_key = ' '.join(tokens[:nkeyfields])
            if test_key == key:
                if nvalfields < 1: # this is a bool
                    val = convert
                elif nvalfields == 1: # this is a scalar
                    val = convert(tokens[nkeyfields])
                else: # this is a vector or tensor
                    val = convert(tokens[nkeyfields:(nkeyfields+nvalfields)])
                if hasattr(configkey, '__iter__'):
                    # In case multiple values are on a single line...
                    for ck, v in zip(configkey, val):
                        if ck is not None:
                            add_info(ck, v)
                else:
                    add_info(configkey, val)
                return True
            else:
                return False

        INFO = 'Info:'
        infile = open(filename, 'r')
        # Skip down to the Info section (lines starting with 'Info:').
        while not infile.readline().startswith(INFO): continue
        for line in infile:
            #     The Info section is over at the first blank line after it 
            # starts; stop reading when that is reached.  Until then, inspect
            # all lines starting with "Info:". See the class definition of 
            # _kvpairs above to see how this is done. 
            #
            if not line.strip(): break

            if line.startswith(INFO):
                tokens = line.lstrip(INFO).strip().split()
                for key, confkey, nfields, dtype in NamdLog._kvpairs:
                    if parse(tokens, key, confkey, nfields, dtype):
                        break
        infile.close()
        if not info:
            raise IOError('Bad NAMD output. Check for errors!')
        return CaseInsensitiveDict(info)

    @staticmethod
    def read_energy(filename):
        """Read the energy entries from a NAMD log file. All types of energy
        outputs are made into a dict containing numpy arrays. These are in turn
        accessible as keywords of a single dict.
        """
        energies = {}
        etag = 'energy'
#        amdtag = 'amd_energy'
        ttag = 'ti'
        TITLE = 'ETITLE:'
        TITITLE = 'TITITLE:'
        FORMAT = 'ENERGY:'
#        AMDFORMAT = 'ACCELERATED MD:'
        TIFORMAT = 'TI:'        
        # standard MD energy log
        #
        energies[etag] = OrderedDict()
        term_indices = {}
        # non-standard energy logs (may or may not exist)
        #
#        energies[amdtag] = OrderedDict()
#        amd_term_indices = {}
        #
        energies[ttag] = OrderedDict()
        ti_term_indices = {}

        terms_are_defined = False
        amd_terms_are_defined = False
        ti_terms_are_defined = False
        for line in open(filename, 'r'):
            # standard MD energy log
            #
            if line.startswith(TITLE) and not terms_are_defined:
                terms_are_defined = True
                terms = line.lstrip(TITLE).strip().split()
                for i, term in enumerate(terms):
                    term_indices[i] = term 
                    energies[etag][term] = []
            elif line.startswith(FORMAT):
                values = line.lstrip(FORMAT).strip().split()
                for i, value in enumerate(values):
                    energies[etag][term_indices[i]].append(float(value))
            # accelerated MD energy log
            #
#            elif line.startswith(AMDFORMAT):
#                terms = line.strip().split()[2::2]
#                values = line.strip().split()[3::2]
#                if not amd_terms_are_defined:
#                    amd_terms_are_defined = True
#                    for i, term in enumerate(terms):
#                        amd_term_indices[i] = term
#                        energie[amdtag][term] = []
#                for i, value in enumerate(values):
#                    idx = amd_term_indices[i]
#                    energies[amdtag][amd_term_indices[i]].append(float(value))
            # TI energy log
            #
            elif line.startswith(TITITLE) and not ti_terms_are_defined:
                ti_terms_are_defined = True
                ti_terms = line.lstrip(TITITLE).strip().split()
                for i, ti_term in enumerate(ti_terms):
                    ti_term_indices[i] = ti_term 
                    energies[ttag][ti_term] = []
            elif line.startswith(TIFORMAT):
                values = line.lstrip(TIFORMAT).strip().split()
                try:
                    for i, value in enumerate(values):
                        energies[ttag][ti_term_indices[i]].append(float(value))
                except (KeyError, ValueError):
                    pass
        for key in energies:
            for term in energies[key]:
                energies[key][term] = asarray(energies[key][term])
        return energies

    @staticmethod
    def read_xgs(filename):
        """Read XGS information from a trajectory.

        WARNING! EXPERIMENTAL! NOT GUARANTEED TO WORK AS EXPECTED!
                 THE XGS OUTPUT FORMAT IS NOT FIXED YET!
        """
        TAG = 'TCL: XGS)'
        xgstype = None
        ladder = None
        weights = None
        indices = []
        for line in open(filename, 'r'):
            if line.startswith(TAG):
                _line = line.lstrip(TAG).strip()
            else:
                continue
            if _line.startswith('simulation type:'):
                xgstype = ' '.join(_line.split()[2:])      
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
        indices = asarray(indices, int32)
        return (xgstype, ladder, weights, indices) 

    @property
    def numsteps(self):
        """The number of MD steps for which output exists. Step 0 may be
        ignored.
        """
        return (self.energy['TS'].size - int(self._nostep0))

    @property
    def temperature(self):
        """The temperature (in Kelvin) set by the _thermostat_. If no
        thermostat is set, this is None, even if 'temperature' is set.
        """
        for key in ('langevinTemp', 'tcoupleTemp', 'rescaleTemp',
                    'reassignTemp', 'loweandersenTemp'):
            try:
                return float(self.info[key])
            except KeyError:
                pass
        return None

    @property
    def kT(self):
        """kT of the simulation using NAMD consistent constants."""
        if self.temperature is not None:
            return NamdLog.BOLTZMANN*self.temperature
        else:
            return None

    @property
    def pressure(self):
        """The pressure (in bar) set by the barostat."""
        for key in ('BerendsenPressureTarget', 'LangevinPistonTarget'):
            try:
                return float(self.info[key])
            except KeyError:
                pass
        return None

    def bond_lambda(self, lambda_=None):
        """Convert the given (array-like) alchLambda(s) to the bonded scaling
        parameter. 

        If None is given, use the alchLambda value that was set in the 
        configuration file.
        """
        if lambda_ is None:
            lambda_ = self.info['alchLambda']
        else:
            lambda_ = asarray(lambda_)
        bond_end = self.info['alchBondLambdaEnd']
        if bond_end > 0.:
            return minimum(1., lambda_ / bond_end)
        else:
            return ones(lambda_.shape)

    def vdw_lambda(self, lambda_=None):
        """Convert the given (array-like) alchLambda(s) to the vdW scaling
        parameter. 

        If None is given, use the alchLambda value that was set in the 
        configuration file.
        """
        if lambda_ is None:
            lambda_ = self.info['alchLambda']
        else:
            lambda_ = asarray(lambda_)
        vdw_end = self.info['alchVdwLambdaEnd']
        if vdw_end > 0.:
            return minimum(1., lambda_ / vdw_end)
        else:
            return ones(lambda_.shape)

    def elec_lambda(self, lambda_=None):
        """Convert the given (array-like) alchLambda(s) to the elec scaling
        parameter. 

        If None is given, use the alchLambda value that was set in the 
        configuration file.
        """
        if lambda_ is None:
            lambda_ = self.info['alchLambda']
        else:
            lambda_ = asarray(lambda_)
        elec_start = self.info['alchElecLambdaStart']
        if elec_start < 1.:
            return maximum(0., (lambda_ - elec_start) / (1. - elec_start))
        else:
            return ones(lambda_.shape)

    def lambdas(self, lambda1=None):
        """Return scaling parameters for all alchemical groups (1 and 2) and 
        interaction types (bond, elec, and vdw) as a tuple:

            (bond1, elec1, vdw1, bond2, elec2, vdw2)

        See:

        bond_lambda
        vdw_lambda
        elec_lambda
        """
        if lambda1 is None:
            lambda1 = self.info['alchLambda']
        else:
            lambda1 = asarray(lambda1)
        lambda2 = 1. - lambda1
        return (self.bond_lambda(lambda1), self.elec_lambda(lambda1),
                self.vdw_lambda(lambda1), self.bond_lambda(lambda2),
                self.elec_lambda(lambda2), self.vdw_lambda(lambda2))

    @property
    def etitle(self):
        """The (formatted) ETITLE string for this MD run."""
        title = ['ETITLE:      TS']
        for term in self.energy:
            title.append(self._fmt_energy_whitespace(term))
            if term != 'TS': 
                title.append(' %14s'%term)
        return ''.join(title)

    def _fmt_energy_whitespace(self, term):
        """Return the extra white space appropriate to a given energy term.
        This is meant as a helper function to aid in string formatting.

        See Controller::printEnergies in Controller.C from the NAMD source code
        for details.
        """
        if term in ('ELECT', 'TOTAL', 'PRESSURE', 'DRUDEBOND', 'GRO_PAIR_LJ',
                    'NATIVE'):
            return '     '
        return ''

    def _fmt_energy_values(self, values, step=0):
        """Return a formatted string for any given set of values that match the 
        energy output (i.e. have the same iteritems() keys).  This is meant as
        a helper function for making custom formatted strings.
        """
        strng = ['ENERGY: %7d'%step]
        for term,energy in values.iteritems():
            strng.append(self._fmt_energy_whitespace(term))
            if term != 'TS': 
                strng.append(' %14.4f'%energy)
        return ''.join(strng)

    def energy_frame(self, step=-1):
        """Return the energy of the given frame as a formatted string."""
        values = OrderedDict()
        for k, v in self.energy.iteritems(): 
            values[k] = v[step]
        return self._fmt_energy_values(values, self.energy['TS'][step])

    def energy_mean(self, start=1, stop=None, step=None):
        """Return the mean of the energy terms as a formatted string.

        The first 'start' frames will be excluded. The default (1) will omit
        step zero.
        """
        values = OrderedDict()
        for k, v in self.energy.iteritems(): 
            values[k] = v[start:stop:step].mean()
        return self._fmt_energy_values(values)

    def energy_diff_mean(self):
        """Return the mean of the differences of sequential energy terms as a
        formatted string.

        i.e. dE(i+1) = E(i+1) - E(i)

        This is useful for checking energy conservation/drift.
        """
        values = OrderedDict()
        for k, v in self.energy.iteritems():
            values[k] = (v[1::2] - v[0::2]).mean()
        return self._fmt_energy_values(values)

    def energy_std(self, start=1, stop=None, step=None, g=None):
        """Return the standard deviation of the energy terms as a formatted 
        string.

        The first 'start' frames will be excluded. The default (1) will omit
        step zero. If the statistical inefficiency, g, is provided, then
        standard errors will be returned instead. That is:

        standard error = sigma / sqrt(N/g)

        where sigma is the standard deviation, N is the number of values, and
        g is the statistical inefficiency.
        """
        if g is None:
            sqrt_Neff_inv = 1.
        else:
            N = self.energy['TS'][start:stop:step].size
            sqrt_Neff_inv = sqrt(float(g) / N)

        values = OrderedDict()
        for k, v in self.energy.iteritems(): 
            values[k] = v[start:stop:step].std(ddof=1)*sqrt_Neff_inv
        return self._fmt_energy_values(values)

    def energy_diff_std(self):
        """Return the standard deviation of the differences of sequential 
        energy terms as a formatted string.
        """
        values = OrderedDict()
        for k, v in self.energy.iteritems():
            values[k] = (v[1::2] - v[0::2]).std(ddof=1)
        return self._fmt_energy_values(values, 0)

#    def amd_energy_mean(self, start=1, stop=None, step=None):
#        """Return the mean of the accelerated MD energy terms as a formatted 
#        string.  The first 'start' frames will be excluded.  The default (1)
#        will omit step zero.
#        """
#        strng = ['ACCELERATED MD: STEP %d'%0]
#        for term, amd_energy in self.amd_energy.iteritems():
#            if term != 'STEP':
#                strng.append(' %s %f'%(term, 
#                                       amd_energy[start:stop:step].mean())
#                            )
#        return ''.join(strng)

    def __eq__(self, other):
        """Check that two NamdLog instances are compatible.

        NB: This only checks that the energy terms in the two files are the
        same. At present, there is no checking of simulation parameters.
        """
        for name in NamdLog._attr_names:
            try:
                selfterms = self.__getattribute__(name).keys()
                selfterms.sort()
            except AttributeError:
                selfterms = None
            try:
                otherterms = other.__getattribute__(name).keys()
                otherterms.sort()
            except AttributeError:
                otherterms = None
            if selfterms != otherterms:
                return False
        return True

    def __iadd__(self, other):
        """Concatenate the energy attributes of one object with another."""
        if not self == other:
            raise TypeError('Mismatch in terms. NamdLogs are incompatible!')
        # Since step 0 is degenerate with step -1, it is usually skipped 
        # (unless it is the _only_ step).
        #
        n0 = (None if self._nostep0 else 1)
        for name in NamdLog._attr_names:
            try:
                selfattr = self.__getattribute__(name)
                otherattr = other.__getattribute__(name)
            except AttributeError:
                continue
            for key in selfattr:
                attrs = (selfattr[key], otherattr[key][n0:])
                selfattr[key] = concatenate(attrs)
        try:
            key = 'state' 
            self.xgs[key] = concatenate((self.xgs[key], other.xgs[key]))
        except AttributeError:
            pass
        return self

