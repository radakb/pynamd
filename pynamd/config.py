"""Basic class for interacting with NAMD configuration files

This is all essentially deprecated after the addition of a (nearly) full Python
interpreter in NAMD. However, select parts of this remain useful in certain
instances, for example, this is probably the easiest way to rebuild a 
configuration file from MD output.
"""
from __future__ import print_function

from pynamd.log import NamdLog


class NamdConfig(object):
    """The NamdConfig class provides rudimentary read/write access to NAMD
    configuration files. At its core this reduces such files to a list of lines
    (the primary attribute of the object) and identifies NAMD settings as
    keyword/value pairs. The main methods provide a thin wrapper to list
    methods such as insert and remove, except that now matching can be done by
    the NAMD keyword alone (as if the object were a dict). Note that
    configuration files are effectively Tcl scripts, so values are returned
    wholesale, as-is - interpreting values is thus left to the user. However,
    querying current settings and/or changing them is possible.

    As in NAMD, keywords are case insensitive. However, unlike NAMD the current
    code will get tricked by multiple specifications with different
    capitalizations (e.g. 'bincoordinates' and 'binCoordinates' will be invalid
    in NAMD, but confusingly valid here).
    """
    def __init__(self, lines=[]):
        self.lines = list(lines)

    def __str__(self):
        """Return the current file contents as a string."""
        return ''.join(self.lines)

    @staticmethod
    def _parse(line):
        """Parse a line.

        Remove comments and return a keyword/value pair as a tuple. This
        separation may be done by any combination of whitespace and equal
        signs. The value may be multiple whitespace separated entries. Empty
        lines return (None, None).

        Example:

        key = val1 val2
        key= val1 val2
        key =val1 val2
        key=val1 val2
        key val1 val2; # a comment        

        will all be parsed as:

        ('key','val1 val2')

        The value element may be a number (e.g. 10), a Tcl variable referred to
        by value (e.g. $n), or a Tcl expression (e.g. [expr {20/2}]). Handling 
        these values as anything other than a string would effectively require
        re-implementation of the Tcl interpreter, so nothing is done beyond
        allowing viewing, adding/deleting, or replacing keyword/value pairs.
        """
        _line = line.split('#')[0].strip().rstrip(';')
        if _line:
            key = _line.split('=')[0].split(' ')[0].rstrip()
            val = _line.lstrip(key).lstrip().lstrip('=').lstrip()
            return key, val
        else:
            return None, None

    def index(self, key):
        """Return the index (or list of indices) of lines that match 'key'. 

        It is an error if there is no such keyword.
        """
        _key = str(key)
        errmsg = "Keyword '%s' is not in the file."%_key
        if _key not in NamdLog._multivals:
            for i, line in enumerate(self.lines):
                kw = self._parse(line)[0] # may be (None, None)
                if str(kw).lower() == _key.lower():
                    return i
            raise KeyError(errmsg)
        else:
            indices = [i for i, line in enumerate(self.lines)
                       if self._parse(line)[0] == _key]
            if indices:
                return indices
            else:
                raise KeyError(errmsg)

    def __getitem__(self, key):
        """Return a value (or list of values) corresponding to the keyword."""
        if key not in NamdLog._multivals:
            return self._parse(self.lines[self.index(key)])[1]
        else:
            return [self._parse(self.lines[i])[1] for i in self.index(key)]
    
    def insert(self, j, key, value):
        """Insert a new keyword/value pair on the "j"th line."""
        self.lines.insert(j, '%s %s\n'%(str(key),str(value)))

    def append(self, key, value):
        """Alias for self.insert(len(self.lines), key, value)."""
        self.insert(len(self.lines), key, value)

    def prepend(self, key, value):
        """Alias for self.insert(0, key, value)."""
        self.insert(0, key, value)

    def remove(self, key, i=0, noexception=False):
        """Remove the "i"th instance of a keyword/value pair.

        If i < 0, remove _all_ instances. Unless 'noexception' is True it is an
        error if there is no such keyword or not enough instances.
        """
        _key = str(key)
        try:
            index = self.index(_key)
        except KeyError:
            if noexception:
                return
            else:
                raise
        if _key not in NamdLog._multivals:
            del self.lines[index]
        else:
            if i < 0:
                for j in range(self.count(_key)):
                    del self.lines[index[j]]
            else:
                del self.lines[index[i]]
            
    def pop(self, key, i=0):
        """Remove and return the "i"th instance of a keyword/value pair.

        It is an error if there is no such keyword or not enough instances.
        """
        if key not in NamdLog._multivals:
            value = self[key]
        else:
            value = self[key][i]
        self.remove(key, i)
        return value 

    def count(self, key):
        """Return the number of times the keyword appears."""
        try:
            return len(self.index(key))
        except TypeError:
            return 1

    def reset(self, key, value, i=0, noexception=False):
        """Set a new value for the "i"th instance of a keyword.

        Unless 'noexception' is True it is an error if there is no such keyword
        or not enough instances.
        """
        _key = str(key)
        try:
            index = self.index(_key)
        except KeyError:
            if noexception:
                return
            else:
                raise
        if _key not in NamdLog._multivals:
            self.lines[index] = '%s %s\n'%(_key, str(value))
        else:
            self.lines[index[i]] = '%s %s\n'%(_key, str(value))

    def reset_or_insert(self, key, value, j=0, i=0):
        """Attempt to reset the "i"th instance of a keyword to the given value.
        If it does not exist, insert the keyword/value pair at the "j"th line
        instead.
        """
        try:
            self.reset(key, value, i)
        except KeyError:
            self.insert(j, key, value)

    def reset_or_append(self, key, value, i=0):
        """Attempt to reset the "i"th instance of a keyword to the given value.
        If it does not exist, append the keyword/value pair instead.

        same as self.reset_or_insert(key, value, len(self.lines), i)
        """
        try:
            self.reset(key, value, i)
        except KeyError:
            self.append(key, value)

    def reset_or_prepend(self, key, value, i=0):
        """Attempt to reset the "i"th instance of a keyword to value. If it
        does not exist, prepend the keyword/value pair instead.

        same as self.reset_or_insert(key, value, 0, i)
        """
        try:
            self.reset(key, value, i)
        except KeyError:
            self.prepend(key, value)
            
    @classmethod
    def read(cls, filename):
        """Return a new object from a NAMD config file."""
        obj = cls()
        obj.lines = open(str(filename), 'r').readlines()
        return obj
 
    @classmethod
    def fromnamdlog(cls, logfilename):
        """Read the standard output from a NAMD run (assumed to be either a 
        file or a NamdLog instance) and re-capitulate the options used in that
        run. The results are returned as a new NamdConfig object.
        """
        # We just need the run info, so we can skip the (potentially expensive)
        # parsing of the energy output.
        #
        if isinstance(logfilename,NamdLog):
            info = logfilename.info
        else:
            info = NamdLog.read_info(logfilename)
        def bool2str(x): return 'on' if x else 'off'
        def list2str(x): return ' '.join(str(xi) for xi in x)
        obj = cls()
        # Iterating the keywords using the _kvpairs tuple assures a fixed,
        # sensible order for parameter outputting, whereas info is a somewhat
        # randomly ordered dict (would an OrderedDict help?).
        #
        for key, configkey, nfields, dtype in NamdLog._kvpairs:
            try:
                if hasattr(configkey, '__iter__'):
                    for ck in configkey:
                        if ck is not None:
                            obj.reset_or_append(ck, info[ck])
                else:
                    if configkey not in NamdLog._multivals:
                        if nfields < 1:
                            value = bool2str(info[configkey])
                        elif nfields > 1:
                            value = list2str(info[configkey])
                        else:
                            value = info[configkey]
                        # Using reset_or_append avoids double setting an option
                        # that has multiple reporting formats.  For example 
                        # 'cutoff' appears differently w/ and w/o 'switching'
                        # set on.
                        obj.reset_or_append(configkey, value)
                    else:
                        # Set these options multiple times (hence no reset).
                        for value in info[configkey]:
                            obj.append(configkey, value)
            except KeyError:
                pass
        return obj

    def write(self, filename, mode='w'):
        """Write a new NAMD config file with the current settings."""
        if hasattr(filename, 'write'):
            outfile = filename
            own_this_handle = False
        else:
            outfile = open(str(filename), mode)
            own_this_handle = True
        outfile.write(str(self))
        if own_this_handle:
            outfile.close()


if __name__ == '__main__':
    """This is a convenient, but by no means exhaustive, set of tests."""
    import sys
    import argparse
    import tempfile
    import subprocess

    def look_for_keyword(kw):
        """Run a simple keyword test. Return True if successful, else 
        return False.
        """
        print('Looking for first "%s" keyword:'%kw)
        print(">>index = conf.index('%s')"%kw)
        try:
            index = conf.index(kw)
            ninst = conf.count(kw)
            try:
                index = index[0]
            except TypeError:
                pass

            print('Found %d instance(s)'%ninst)
            print('First instance is on line %d:'%index)
            print(conf.lines[index])
            return True
        except ValueError:
            print('None found!')
            return False

    def reset_keyword(kw):
        """Try to replace a keyword with a bunk value."""
        print("Resetting first instance of keyword '%s' to'foo'"%kw)
        print('>>conf[%s]'%kw)
        print(conf[kw])
        print(">>conf.reset(%s, 'foo')"%kw)
        try:
            conf.reset(kw, 'foo')
            print('>>conf[%s]'%kw)
            print(conf[kw])
        except:
            print('Failed!')
        print()

    def remove_keyword(kw):
        """Try to remove a keyword."""
        print("Removing first instance of keyword '%s'"%kw)
        print(">>conf.remove(%s)"%kw)
        conf.remove(kw)
        try:
            conf[kw]
            print('Failed!')
        except:
            print('Success!')
        print()

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, help='NAMD configuration file') 
    parser.add_argument('--kw', type=str, nargs='*', default=None,
                        help='Keywords to test for.')
    parser.add_argument('--log', type=str, default=None)
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if args.conf is not None:
        print('Testing object construction from configuration file:')
        print('>>conf = NamdConfig.read(%s)'%args.conf)
        conf = NamdConfig.read(args.conf)
        print()
    if args.log is not None:
        print('Testing object construction from NamdLog file:')
        print('>>conf = NamdConfig.fromnamdlog(%s)'%args.log)
        conf = NamdConfig.fromnamdlog(args.log)
        print()

    if args.kw is None:
        args.kw = ['structure', 'cutoff']
    exists = [look_for_keyword(kw) for kw in args.kw]
    
    print('Testing parameter resetting')
    [reset_keyword(kw) for kw, e in zip(args.kw, exists) if e]
   
    print('Testing parameter removal')
    [remove_keyword(kw) for kw, e in zip(args.kw, exists) if e]

    print('Testing parameter clear of parameters and binvelocities')
    conf.remove('parameters', -1, True)
    conf.remove('binvelocities', -1, True)

    if args.conf is not None:
        print('writing a new configuration to a tempfile and diffing')
        tmp = tempfile.NamedTemporaryFile()
        print('conf.write(<temporary file>)')
        conf.write(tmp.name)
        print()
        print('results of diff:')
        subprocess.call(['/usr/bin/env', 'diff', args.conf, tmp.name])
        tmp.file.close()

    conf.reset('extendedsystem', 'foo.xsc', noexception=True)
