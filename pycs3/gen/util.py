"""
Various useful stuff.
For now there are some wrappers to pickle objects.
"""
import csv
import datetime
import glob
import gzip
import logging
import math
import os
import pickle as pickle

import numpy as np

import pycs3

logger = logging.getLogger(__name__)

tracei = 1  # global variable, filename to write trace pkl.


def writepickle(obj, filepath, verbose=True, protocol=-1):
    """
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
    if os.path.splitext(filepath)[1] == ".gz":
        pkl_file = gzip.open(filepath, 'wb')
    else:
        pkl_file = open(filepath, 'wb')

    pickle.dump(obj, pkl_file, protocol)
    pkl_file.close()
    if verbose:
        logger.info("Wrote %s" % filepath)


def readpickle(filepath, verbose=True):
    """
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
    if os.path.splitext(filepath)[1] == ".gz":
        pkl_file = gzip.open(filepath, 'rb')
    else:
        pkl_file = open(filepath, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    if verbose:
        logger.info("Read %s" % filepath)
    return obj


def readidlist(filepath, verbose=True):
    """
	Reads a textfile with "one point per line", probably a skiplists.
	Accepts blank lines, and lines starting with # will not be read.
	Format of the lines is : id [comment]
	If this is a skiplist, id is a MJD.
	"""

    if not os.path.exists(filepath): # pragma: no cover
        raise RuntimeError("File does not exist : %s" % filepath)

    myfile = open(filepath, "r")
    lines = myfile.readlines()
    myfile.close()
    table = []
    for line in lines:
        if line[0] == '#' or len(line) < 4:
            continue

        if len(line.split()) > 1:
            id = line.split()[0]
            comment = line.split(None, 1)[1:][0].rstrip('\n')
        else:
            id = line.split()[0]
            comment = ""

        table.append([id, comment])
    if verbose:
        logger.info("I've read %i lines from %s" % (len(table), os.path.basename(filepath)))
    return table


def trace(lclist=[], splist=[], tracedir="trace"):
    """
	Function to save a "trace" of processes modifying lightcurves and splines, like optimizers do. 
	Just call this from inside your loop etc, I will save your current lightcurves and splines into a pickle inside the tracedir.
	I increment the filenames.
	
	The argument "level" is about what should be saved.
	level = "Full" : Everything you give me is saved in the pickle. Now this is large ...
	level = "Light" : I try to reduce filesize of the pickle, by removing the splines datapoints etc. You can still plot these objects.
	"""
    if not os.path.exists(tracedir):
        os.mkdir(tracedir)
    global tracei
    filepath = os.path.join(tracedir, "%06i.pkl" % tracei)
    if os.path.exists(filepath): # pragma: no cover
        raise RuntimeError("Sorry, I don't want to overwrite the existing trace inside '%s'." % tracedir)

    now = datetime.datetime.now()
    writepickle({"lclist": lclist, "splist": splist, "datetime": now}, filepath, verbose=True, protocol=-1)
    tracei += 1


def plottrace(tracedir="trace", reset=False, showspl=True, **kwargs):
    """
	Turns a trace into plots ...
	reset = True : I will remove all shifts/ML etc, just show the real "observations".
	kwargs are passed to the display function.
	"""

    tracepkls = glob.glob(os.path.join(tracedir, "??????.pkl"))

    def plot(tracepkl):
        pkl = readpickle(tracepkl, verbose=True)
        if reset:
            for l in pkl["lclist"]:
                l.timeshift = 0.0
                l.magshift = 0.0
                l.fluxshift = 0.0
                l.ml = None
        if not showspl:
            pkl["splist"] = []

        shiftstxt = "(%s)" % "/".join(["%+.1f" % (getattr(l, "truetimeshift", 0.0)) for l in pkl["lclist"]])
        titletxt = "%s %s %s" % (tracedir, "", shiftstxt)
        pycs3.gen.lc_func.display(pkl["lclist"], pkl["splist"], title=titletxt, filename=tracepkl[:-3] + ".png", **kwargs)

    map(plot, tracepkls)


def multilcsexport(lclist, filepath, separator="\t", rdbunderline=True, verbose=True, properties=None, keep_masks=False):
    """
    Writes the a list of lightcurves as flat acscii files into one single file.
    Normally you should prefer writing each lightcurve into a single file, using
    :py:meth:`pycs3.gen.lc.LightCurve.rdbexport`.

    Note that only lightcurves of same length and sampling can be written with this function !

    :param lclist: A list of lightcurve objects to write
    :type lclist: list
    :param filepath: where to write
    :type filepath: str
    :param separator: how to separate the collumns
    :type separator: str
    :param rdbunderline: do you want the "=====" underlining ?
    :type rdbunderline: bool
    :param properties: properties of the lightcurves to include in the file.
    :type properties: list of strings
    :param verbose: verbosity
    :type verbose: bool
    :param keep_masks: do you want to add the flags in the output file
    :type keep_masks: bool
    """

    # We start with a few tests to see if it is possible to write these lcs into a single file ...
    commonjds = lclist[0].getjds()
    lenlc = len(lclist[0])

    for thislc in lclist:
        thislc.validate()  # Good idea to keep this here, as the code below is so ugly ...
        if len(thislc) != lenlc: # pragma: no cover
            raise RuntimeError("Lightcurve %s has not the same length !" % str(thislc))

        if not np.allclose(thislc.getjds(), commonjds, rtol=0.0, atol=1e-5):# pragma: no cover
            raise RuntimeError("Lightcurve %s has different epochs !" % str(thislc))

    # Now we check the properties. At least a minimal check : they should be available for all the
    # lightcurves.
    if properties is None:
        properties = []

    for property in properties:
        for l in lclist:
            if not property in l.commonproperties():# pragma: no cover
                raise RuntimeError("Lightcurve %s has no property %s" % (l, property))

            # We also have to check that all those properties are identical for all lcs !
        firstprops = [p[property] for p in lclist[0].properties]
        for l in lclist:
            if not firstprops == [p[property] for p in l.properties]:# pragma: no cover
                raise RuntimeError("Properties not identical !")

    # Ok, now we prepare the data to write into that file.
    colnames = []
    data = []

    colnames.append("mhjd")
    data.append(["%.5f" % commonjd for commonjd in commonjds])

    for thislc in lclist:
        colnames.append("mag_" + thislc.object)
        # data.append(["%09.5f" % mag for mag in thislc.getmags()])
        data.append(["%.5f" % mag for mag in thislc.getmags()])
        colnames.append("magerr_" + thislc.object)
        data.append(["%.5f" % magerr for magerr in thislc.magerrs])

    # And now the properties
    for property in properties:
        values = [p[property] for p in lclist[0].properties]
        colnames.append(property)
        data.append(values)

    if keep_masks:
        colnames.append('flag')
        flags = lclist[0].mask
        for thislc in lclist:
            assert np.all(thislc.mask == flags), 'The mask of your light curves does not match'
        data.append(flags)

    # We put all this together :
    datatransposed = list(zip(*data))  # Yep !
    rdbunderlines = ["=" * len(colname) for colname in colnames]
    if rdbunderline:
        biglist = [colnames, rdbunderlines]
    else:
        biglist = [colnames]
    biglist.extend(datatransposed)

    # biglist now contains the file items line by line.

    # we write the file
    csvwriter = csv.writer(open(filepath, 'w'), delimiter=separator, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerows(biglist)
    if verbose:
        logger.info("Wrote the lightcurves into %s" % filepath)


def datetimefromjd(JD):
    """
	Copy and past from cosmouline.
	Can be of use here to plot lightcurves with nice dates.
	
	Returns the Gregorian calendar (i.e. our "normal" calendar)
	Based on wikipedia:de and the interweb :-)
	
	
	:type JD: float
	:param JD: julian date

	:rtype: datetime object
	:returns: corresponding datetime
	
	
	"""

    if JD < 0:# pragma: no cover
        raise ValueError('Julian Day must be positive')

    (F, Z) = math.modf(JD + 0.5)
    Z = int(Z)

    if JD < 2299160.5:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day = B - D - int(30.6001 * E) + F
    nday = B - D - 123
    if nday <= 305:
        dayofyr = nday + 60
    else:
        dayofyr = nday - 305
    if E < 14:
        month = E - 1
    else:
        month = E - 13

    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    # a leap year?
    leap = 0
    if year % 4 == 0:
        leap = 1

    if year % 100 == 0 and year % 400 != 0:
        leap = 0
    if leap and month > 2:
        dayofyr = dayofyr + leap

    # Convert fractions of a day to time
    (dfrac, days) = math.modf(day / 1.0)
    (hfrac, hours) = math.modf(dfrac * 24.0)
    (mfrac, minutes) = math.modf(hfrac * 60.0)
    seconds = round(mfrac * 60.0)  # seconds are rounded

    if seconds > 59:
        seconds = 0
        minutes = minutes + 1
    if minutes > 59:
        minutes = 0
        hours = hours + 1
    if hours > 23:
        hours = 0
        days = days + 1

    return datetime.datetime(year, month, int(days), int(hours), int(minutes), int(seconds))


def strtd(td):
    """
	To print out time differences ...
	Could be improved a bit :-)
	"""
    strdiff = str(td)  # looks like 0:02:04.43353
    return strdiff.split(".")[0]