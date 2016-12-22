import pynamd
import pytest
import numpy as np

#get local test directory
import os
DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def test_incorrect_input():
    """Expects to raise and IOError"""
    with pytest.raises(IOError):
        pynamd.NamdLog(DIR + "data/00002-err.log")

def test_minimal_energy_parsing():
    log = pynamd.NamdLog(DIR + "data/minimal.log")        
    assert np.all(log.energy['TS'] == [0, 2500, 5000, 7500, 10000])
    assert np.all(log.energy['TOTAL'] == [-150044.8014, -149942.7782, -149771.7871, -149754.5072, -150136.1524])
    assert np.all(log.energy['TEMP'] == [297.767 ,  299.4523,  298.2334,  297.5036,  299.8304])

def test_minimal_info_parsing():
    log = pynamd.NamdLog(DIR + "data/minimal.log")        
    assert log.info['timestep'] == 2