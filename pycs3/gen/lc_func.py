"""
Module containing the function related to LightCurve operation
It contains light curves plotting routine, and all function that operate on a *list* of LightCurve
"""
import numpy as np
import os
from pycs3.gen.lc import LightCurve
from pycs3.gen.util import datetimefromjd
# For the sort and shuffle stuff :
import random
import operator


# Factory functions : give me some stuff and I make a lightcurve object from this.
def factory(jds, mags, magerrs=None, telescopename="Unknown", object="Unknown", verbose=False):
    """Returns a valid lightcurve object from the provided arrays.
    The numpy arrays jds and mags are mandatory. If you do not specify a third array containing the magerrs,
    we will calculate them "automatically" (all the same value), to avoid having 0.0 errors.

    :type	jds: 1D numpy array
    :param	jds: julian dates
    :type	mags: 1D numpy array
    :param	mags: magnitudes
    :type	magerrs: 1D numpy array
    :param	magerrs: optional magnitude errors
    :type telescopename: string
    :param telescopename: name of the telescope
    :type object: string
    :param object: name of the astronomical object
    :type verbose: bool
    :param verbose: verbosity of the function
    """

    # Make a brand new lightcurve object :
    newlc = LightCurve()

    # Of couse we can/should check a lot of things, but let's be naive :

    newlc.jds = np.asarray(jds)
    newlc.mags = np.asarray(mags)

    if magerrs is None:
        newlc.magerrs = np.zeros(len(newlc.jds)) + 0.1
    else:
        newlc.magerrs = np.asarray(magerrs)

    if len(newlc.jds) != len(newlc.mags) or len(newlc.jds) != len(newlc.magerrs):
        raise RuntimeError("lightcurve factory called with arrays of incoherent lengths")

    newlc.mask = newlc.magerrs >= 0.0  # This should be true for all !

    newlc.properties = [{}] * len(newlc.jds)

    newlc.telescopename = telescopename
    newlc.object = object

    newlc.setindexlabels()
    newlc.commentlist = []

    newlc.sort()  # not sure if this is needed / should be there

    newlc.validate()

    if verbose:
        print("New lightcurve %s with %i points" % (str(newlc), len(newlc.jds)))

    return newlc


def flexibleimport(filepath, jdcol=1, magcol=2, errcol=3, startline=1, flagcol=None, propertycols=None,
                   telescopename="Unknown", object="Unknown", plotcolour="crimson", verbose=True, absmagerrs=False):
    """
    The general form of file reading. We read only one lightcurve object.
    To comment a line in the input file, start the line with # like in python.

    :param filepath: path to the file to be imported
    :param jdcol: The column number containing the MHJDs. First column is number 1, not 0 !
    :param magcol: The column number containing the magnitudes.
    :param errcol: The column number containing the magnitude errorbars.
    :param flagcol: (default : None) The colum containing a mask (if available). This one should contain False (= will be masked) or True (not masked).
    :param startline: Line from where to start (1 will skip the header line)
    :param telescopename: name of the telescope
    :param object: name of the astronomical object
    :param plotcolour: colour of the lightcurve
    :param verbose: verbosity of the function
    :param absmagerrs: boolean, allow to take the absolute value of the given errors
    :param propertycols: (default : None) is a dict : ``propertycols = {"fwhm":7, "skylevel":8}`` means that col 7 should be read in as property "fwhm" and 8 as "skylevel".
    :type propertycols: dictionary


    """
    if verbose:
        print("Reading \"%s\"..." % (os.path.basename(filepath)))
    rdbfile = open(filepath, "r")
    rdbfilelines = rdbfile.readlines()[startline - 1:]  # we directly "skip" the first lines of eventual headers ...
    rdbfile.close()

    jds = []
    mags = []
    magerrs = []
    flags = []
    properties = []

    elsperline = None

    for i, line in enumerate(rdbfilelines):

        if line[0] == "#":
            continue

        if len(line.strip()) < 5:
            print("Skipping empty line %i : %s" % (i + startline, repr(line)))
            continue

        elements = line.split()  # line is a string, elements is a list of strings

        # We check the consistency of the number of elements...
        if elsperline is not None:
            if len(elements) != elsperline:
                raise RuntimeError("Parsing error in line %i, check columns : \n%s" % (i + startline, repr(line)))
        elsperline = len(elements)

        jds.append(float(elements[jdcol - 1]))
        mags.append(float(elements[magcol - 1]))
        magerrs.append(float(elements[errcol - 1]))

        if flagcol is not None:
            strflag = str(elements[flagcol - 1])
            if strflag == "True":
                flags.append(True)
            elif strflag == "False":
                flags.append(False)
            else:
                print("Flag error in line %i : %s" % (i + startline, repr(line)))
                flags.append(True)
        else:
            flags.append(True)

        propdict = {}  # an empty dict for now
        if propertycols is not None:
            for (propname, propcol) in list(propertycols.items()):
                propdict[propname] = str(elements[propcol - 1])
        properties.append(propdict)

    if absmagerrs:
        magerrs = np.abs(np.array(magerrs))

    # Make a brand new lightcurve object :
    newlc = factory(np.array(jds), np.array(mags), magerrs=np.array(magerrs), telescopename=telescopename,
                    object=object)
    newlc.properties = properties[:]
    newlc.mask = np.array(flags[:])
    newlc.plotcolour = plotcolour
    nbmask = np.sum(newlc.mask == False)
    commentcols = "(%i, %i, %i)" % (jdcol, magcol, errcol)
    newlc.commentlist.insert(0, "Imported from %s, columns %s" % (os.path.basename(filepath), commentcols))
    if verbose:  # noinspection PyStringFormat
        print("%s with %i points imported (%i of them masked)." % (str(newlc), len(newlc.jds), nbmask))
    return newlc


def rdbimport(filepath, object="Unknown", magcolname="mag", magerrcolname="magerr", telescopename="Unknown",
              plotcolour="red", mhjdcolname="mhjd", flagcolname=None, propertycolnames="lcmanip", verbose=True,
              absmagerrs=False):
    """
    The relaxed way to import lightcurves, especially those from cosmouline, provided they come as rdb files.
    Don't care about column indices, just give the column headers that you want to read.

    Propertycolnames is a list of column names that I will add as properties.
    Possible settings :
    "lcmanip" : (default) I will import the standard cols from lcmanip / cosmouline, if those are available.
    None : I will not import any properties
    ["property1", "property2"] : just give a list of properties to import.

    The default column names are those used by the method :py:meth:`pycs.gen.lc.lightcurve.rdbexport` : "mhjd", "mag", "magerr", "mask".

    We use flexibleimport under the hood.

    """

    if verbose:
        print("Checking header of \"%s\"..." % (os.path.basename(filepath)))
    rdbfile = open(filepath, "r")
    rdbfilelines = rdbfile.readlines()
    rdbfile.close()
    headers = rdbfilelines[0].split()
    underlines = rdbfilelines[1].split()
    if list(map(len, headers)) != list(map(len, underlines)):
        raise RuntimeError("Error in parsing headers")
    # headerindices = np.array(range(len(headers))) + 1 # +1 as we use human convention in flexibleimport

    # We build the default property names :
    if propertycolnames == "lcmanip":  # Then we put it the default set, but only if available.
        propertycolnames = set(
            ["telescope", "fwhm", "ellipticity", "airmass", "relskylevel", "normcoeff", "nbimg"]).intersection(
            set(headers))

    # We check if the headers you want are available :
    checknames = [mhjdcolname, magcolname, magerrcolname]
    if flagcolname:
        checknames.append(flagcolname)
    if propertycolnames:
        checknames.extend(propertycolnames)
    for name in checknames:
        if name not in headers:
            raise RuntimeError('I cannot find a column named "%s" in your file !' % name)

    jdcol = headers.index(mhjdcolname) + 1  # +1 as we use human convention in flexibleimport
    magcol = headers.index(magcolname) + 1
    errcol = headers.index(magerrcolname) + 1

    if flagcolname is not None:
        flagcol = headers.index(flagcolname) + 1
    else:
        flagcol = None

    if propertycolnames is not None:
        propertycols = {}
        for propertycolname in propertycolnames:
            propertycols[propertycolname] = headers.index(propertycolname) + 1
    else:
        propertycols = None

    newlc = flexibleimport(filepath, jdcol=jdcol, magcol=magcol, errcol=errcol, startline=3, flagcol=flagcol,
                           propertycols=propertycols, telescopename=telescopename, object=object, verbose=verbose,
                           absmagerrs=absmagerrs)
    newlc.plotcolour = plotcolour
    return newlc


# noinspection PyDefaultArgument,PyDefaultArgument
def display(lclist=[], splist=[],
            title=None, titlexpos=None, style=None, showlegend=True, showlogo=False, logopos="left",
            showdates=False,
            showdelays=False, nicefont=False, text=None, keeponlygrid=False,
            jdrange=None, magrange=None, figsize=None, plotsize=(0.08, 0.96, 0.09, 0.95), showgrid=False,
            markersize=6, showerrorbars=True, showdatapoints=True, errorbarcolour="#BBBBBB", capsize=3,
            knotsize=0.015,
            legendloc="best", showspldp=False, colourprop=None, hidecolourbar=False, transparent=False,
            show_ylabel=True,
            collapseref=False, hidecollapseref=False, jdmintickstep=100, magmintickstep=0.2, filename="screen",
            showinsert=None, insertname=None, verbose=False, ax=None):
    """
    Function that uses matplotlib to plot a **list** of lightcurves/splines/GPRs, either on screen or into a file.
    It uses lightcurve attributes such as ``lc.plotcolour`` and ``lc.showlabels``, and displays masked points
    as black circles. It's certainly a key function of pycs.
    You can also put tuples (lightcurve, listofseasons) in the lclist, and the seasons will be drawn.
    This function is intended both for interactive exploration and for producing publication plots.

    :param show_ylabel:
    :param hidecollapseref:
    :param lclist: A list of lightcurve objects [lc1, lc2, ...] you want to plot.
    :type lclist: list

    :param splist: A list of spline or rslc (e.g., GPR) objects to display on top of the data points.
    :type splist: list

    :param title: Adds a title to the plot, center top of the axes, usually used for lens names.
        To nicely write a lens name, remember to use raw strings and LaTeX mathrm, e.g. :
        ``title = r"$\mathrm{RX\,J1131-1231}$"``
    :type title: string If you want to use a preset of parameter. Choose between "homepagepdf", "homepagepdfnologo",
    "2m2", "posterpdf", "cosmograil_dr1", "cosmograil_dr1_microlensing", I overwrite other keyword arguments.

    :param titlexpos: Specify where you want your to anchor the center of your title in the x axis. the value is in the x-axis fraction.  Default is center. (x = 0.5)
    :type titlexpos: float

    :param style: A shortcut to produce specific kinds of stylings for the plots.
        Available styles:

            * ``homepagepdf`` and "homepagepdfnologo" : for cosmograil homepage, ok also with long magnitude labels (like -13.2)
            * ``2m2`` : for 2m2 data
            * ``posterpdf`` : for 2m2 data
            * ``internal`` : for 2m2 data

    :type style: string

    :param showlegend: Automatic legend (too technical/ugly for publication plots, uses str(lightcurve))
    :type showlegend: boolean

    :param showlogo: Adds an EPFL logo + www.cosmograil.org on the plot.
    :type showlogo: boolean

    :param logopos: Where to put it, "left" or "right"
    :type logopos: string

    :param showdates: If True, the upper x axis will show years and 12 month minor ticks.
    :type showdates: boolean

    :param showdelays: If True, the relative delays between the curves are written on the plot.
    :type showdelays: boolean

    :param nicefont: Sets default to serif fonts (terrible implementation, but works)
    :type nicefont: boolean

    :param text:
        Generic text that you want to display on top of the plot, in the form : [line1, line2, line3 ...]
        where line_i is (x, y, text, kwargs) where kwargs is e.g. {"fontsize":18} and x and y are relative positions (from 0 to 1).
    :type text: list

    :param jdrange: Range of jds to plot, e.g. (53000, 54000).
    :type jdrange: tuple

    :param magrange: Range of magnitudes to plot, like ``magrange = [-11, -13]``.
        If you give only a float, like ``magrange=1.0``, I'll plot +/- this number around the mean curve level
        Default is None -> automatic mag range.
    :type magrange: tuple

    :param figsize: Figure size (width, height) in inches, for interactive display or savefig.
    :type figsize: tuple

    :param plotsize: Position of the axes in the figure (left, right, bottom, top), default is (0.065, 0.96, 0.09, 0.95).
    :type plotsize: tuple

    :param showgrid: Show grid, that is vertical lines only, one for every year.
    :type showgrid: boolean

    :param markersize: Size of the data points, default is 6
    :type markersize: float

    :param showerrorbars: If False, the ploterrorbar settings of the lightcurves are disregared and no error bars are shown.
    :type showerrorbars: boolean

    :param showdatapoints: If False, no data points are shown. Useful if you want e.g. to plot only the microlensing
    :type showerrorbars: boolean

    :param keeponlygrid: If True, keeps the yearly grid from showdates but do not display the dates above the plot.
    :type keeponlygrid: boolean

    :param showinsert: If True, display the insertname image in the top-right corner of the main image
    :type showinsert: boolean

    :param insertname: path to the image you want to insert
    :param errorbarcolour: Color for the error bars
    :type errorbarcolour: string

    :param capsize: Size of the error bar "ticks"
    :type capsize: float

    :param knotsize: For splines, the length of the knot ticks, in magnitudes.
    :type knotsize: float

    :param legendloc: Position of the legend. It is passed to matplotlib legend. It can be useful to specify this if you want to make animations etc.

            * string	int
            * upper right	1
            * upper left	2
            * lower left	3
            * lower right	4
            * right	5


            * center left	6
            * center right	7
            * lower center	8
            * upper center	9
            * center	10

    :type legendloc: string or int

    :param showspldp: Show the acutal data points of spline objects (for debugging etc)
    :type showspldp: boolean

    :param colourprop: If this is set I will make a scatter plot with points coloured according to the given property, disregarding the lightcurve attribute plotcolour.
        Format : (property_name, display_name, minval, maxval), where display_name is a "nice" version of property_name, like "FWHM [arcsec]" instead of "seeing".
        Note that this property will be used in terms of a float. So you cannot use this for properties that are not floats.
    :type colourprop: tuple

    :param hidecolourbar: Set to True to hide the colourbar for the colourprop
    :type hidecolourbar: boolean

    :param transparent: Set transparency for the plot, if saved using filename
    :type transparent: boolean

    :param collapseref: Plot one single dashed line as the reference for the microlensing.
        Use this if you would otherwise get ugly overplotted dashed lines nearly at the same level ...
        This option is a bit ugly, as it does not correct the actual microlensing curves for the collapse.
    :type collapseref: boolean

    :param jdmintickstep: Minor tick step for jd axis
    :type jdmintickstep: float

    :param magmintickstep: Minor tick step for mag axis
    :type magmintickstep: float

    :param filename: If this is not "screen", I will save the plot to a file instead of displaying it. Try e.g. "test.png" or "test.pdf". Success depends on your matplotlib backend.
    :type filename: string

    :param ax: if not None, I will return what I plot in the given matplotlib axe you provide me with instead of plotting it.
    :type ax: matplotlib axes

    :param verbose: Set to True if you want me to print some details while I process the curves
    :type verbose: boolean


    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import MultipleLocator
    import matplotlib.dates
    import matplotlib.lines

    if style is None:
        pass
    elif style in ["homepagepdf", "homepagepdfnologo"]:
        if figsize is None:
            figsize = (10, 5)
        plotsize = (0.09, 0.97, 0.10, 0.95)
        showlogo = True
        if style == "homepagepdfnologo":
            showlogo = False
        nicefont = True
        showdelays = False
        showlegend = False
        showdates = True
        errorbarcolour = "#777777"
        markersize = 5.0
        capsize = 0
        jdmintickstep = 50
        magmintickstep = 0.2
        showgrid = True
        transparent = False
        show_ylabel = True

    elif "2m2" in style:
        if figsize is None:
            figsize = (10, 5)
        plotsize = (0.09, 0.97, 0.10, 0.95)
        showlogo = True
        if "nologo" in style:
            showlogo = False
        logopos = "right"
        nicefont = True
        showdelays = False
        if "showdelays" in style:
            showdelays = True
        showlegend = False
        # showdates=False
        errorbarcolour = "#777777"
        markersize = 7.0
        capsize = 0
        jdmintickstep = 50
        if "largeticks" in style:
            jdtickstep = 100
        magmintickstep = 0.2
        # showgrid=False
        transparent = False
        show_ylabel = True

    elif style == "posterpdf":
        if figsize is None:
            figsize = (10, 5)
        plotsize = (0.09, 0.97, 0.10, 0.95)
        showlogo = False
        nicefont = False
        showdelays = False
        showlegend = False
        showdates = True
        errorbarcolour = "#777777"
        markersize = 4.0
        capsize = 0
        jdmintickstep = 50
        magmintickstep = 0.2
        # showgrid=False
        transparent = False
        show_ylabel = True

    elif style == "internal":
        if figsize is None:
            figsize = (10, 5)
        plotsize = (0.09, 0.97, 0.10, 0.95)
        showlogo = False
        nicefont = True
        showdelays = False
        showdates = True
        errorbarcolour = "#777777"
        markersize = 5.0
        capsize = 0
        jdmintickstep = 50
        magmintickstep = 0.2
        showgrid = True
        transparent = False
        show_ylabel = True

    elif style == "cosmograil_dr1":
        if figsize is None:
            figsize = (10, 5)
        plotsize = (0.09, 0.97, 0.10, 0.95)
        showlogo = False
        nicefont = True
        showdelays = False
        showlegend = False
        showdates = False
        errorbarcolour = "#777777"
        markersize = 5.0
        capsize = 0
        jdmintickstep = 50
        magmintickstep = 0.2
        showgrid = False
        transparent = False
        show_ylabel = True

    elif style == "cosmograil_dr1_microlensing":
        if figsize is None:
            figsize = (10, 5)
        plotsize = (0.09, 0.97, 0.10, 0.95)
        showlogo = False
        nicefont = True
        showdelays = False
        showlegend = False
        showdates = False
        errorbarcolour = "#777777"
        markersize = 5.0
        capsize = 0
        jdmintickstep = 50
        magmintickstep = 0.2
        showgrid = False
        transparent = False
        show_ylabel = False
    else:
        raise RuntimeError("I do not know the style %s" % style)

    if not (isinstance(lclist, list) or isinstance(lclist, tuple)):
        raise TypeError("Hey, give me a LIST of lightcurves !")

    if colourprop is not None:
        (colourpropname, colournicename, colourminval, colourmaxval) = colourprop

    labelfontsize = 14
    if nicefont:
        # mpl.rcParams['font.size'] = 20
        mpl.rcParams['font.family'] = 'serif'
    # labelfontsize = 20
    else:
        labelfontsize = 14

    if ax is None:
        fig = plt.figure(figsize=figsize)  # sets figure size
        fig.subplots_adjust(left=plotsize[0], right=plotsize[1], bottom=plotsize[2], top=plotsize[3])
        axes = plt.gca()

    else:
        axes = ax

    if verbose:
        print("Plotting %i lightcurves and %i splines ..." % (len(lclist), len(splist)))

    reflevels = []  # only used for collapseref

    # The lightcurves :
    for curve in lclist:
        if showdatapoints:
            if type(curve).__name__ == 'tuple':  # then we have both a lightcurve and a season to plot

                actualcurve = curve[0]
                curveseasons = curve[1]

                if not isinstance(curveseasons, list):
                    raise TypeError("lc.display wants LISTs of seasons, not individual seasons !")
                for curveseason in curveseasons:
                    # the x lims :
                    (x1, x2) = curveseason.getjdlims(actualcurve)
                    # for y, lets take the median of that season
                    y = np.median(actualcurve.getmags()[curveseason.indices])

                    # we make this robust even with old versions of matplotlib, so no fancy arrows here.
                    axes.plot([x1, x2], [y, y], color=actualcurve.plotcolour, dashes=(1, 1))
                    axes.annotate(str(curveseason), ((x1 + x2) / 2.0, y), xytext=(-50, -15),
                                  textcoords='offset points',
                                  size=10, color=actualcurve.plotcolour)

                curve = curve[0]  # for the rest of this loop, curve is now only the lightcurve.

            if verbose:
                print("#   %s -> %s\n\t%s" % (curve, str(curve.plotcolour), "\n\t".join(curve.commentlist)))

            tmpjds = curve.getjds()
            tmpmags = curve.getmags()  # to avoid calculating the microlensing each time we need it

            if colourprop is not None:
                scattervalues = np.array([float(propertydict[colourpropname]) for propertydict in curve.properties])
                axes.scatter(tmpjds, tmpmags, s=markersize, c=scattervalues, vmin=colourminval, vmax=colourmaxval,
                             edgecolors="None")
            else:
                if curve.ploterrorbars and showerrorbars:
                    axes.errorbar(tmpjds, tmpmags, curve.magerrs, fmt=".", markersize=markersize,
                                  markeredgecolor=curve.plotcolour, color=curve.plotcolour, ecolor=errorbarcolour,
                                  capsize=capsize, label=str(curve), elinewidth=0.5)
                else:
                    axes.plot(tmpjds, tmpmags, marker=".", markersize=markersize, linestyle="None",
                              markeredgecolor=curve.plotcolour, color=curve.plotcolour, label=str(curve))

            # We plot little round circles around masked points.
            axes.plot(tmpjds[curve.mask == False], tmpmags[curve.mask == False], linestyle="None", marker="o",
                      markersize=8., markeredgecolor="black", markerfacecolor="None", color="black")

        # And now we want to graphically display the microlensing in a nice way. This costs some cpu but anyway
        # for a display it's fine.
        # if curve.ml != None and curve.hideml == False:
        if curve.ml is not None:
            if curve.ml.mltype in ["poly", "leg"]:
                for sfct in curve.ml.mllist:
                    smoothml = sfct.smooth(curve)
                    if not collapseref:
                        axes.plot(smoothml['jds'], smoothml['refmags'], color=curve.plotcolour,
                                  dashes=(3, 3))  # the new ref
                    else:
                        reflevels.append(np.mean(smoothml['refmags']))
                    axes.plot(smoothml['jds'], smoothml['refmags'] + smoothml['ml'], color=curve.plotcolour)
            if curve.ml.mltype == "spline":
                smoothml = curve.ml.smooth(curve)

                if not collapseref:
                    axes.plot(smoothml['jds'], np.zeros(smoothml["n"]) + smoothml['refmag'], color=curve.plotcolour,
                              dashes=(3, 3))  # the new ref
                else:
                    reflevels.append(smoothml['refmag'])

                if hasattr(curve, "hideml"):
                    if not curve.hideml:
                        axes.plot(smoothml['jds'], smoothml['refmag'] + smoothml['ml'], color=curve.plotcolour)
                else:
                    axes.plot(smoothml['jds'], smoothml['refmag'] + smoothml['ml'], color=curve.plotcolour)
                # We want to overplot the knots
                if hasattr(curve, "hideml"):
                    if not curve.hideml:
                        if getattr(curve.ml.spline, "showknots", True):
                            axes.errorbar(smoothml['knotjds'], smoothml['knotmags'] + smoothml["refmag"],
                                          knotsize * np.ones(len(smoothml['knotjds'])), capsize=0,
                                          ecolor=curve.plotcolour, linestyle="none", marker="", elinewidth=1.5)
                else:
                    if getattr(curve.ml.spline, "showknots", True):
                        axes.errorbar(smoothml['knotjds'], smoothml['knotmags'] + smoothml["refmag"],
                                      knotsize * np.ones(len(smoothml['knotjds'])), capsize=0,
                                      ecolor=curve.plotcolour,
                                      linestyle="none", marker="", elinewidth=1.5)

        # Labels if wanted :
        if curve.showlabels:
            for i, label in enumerate(curve.labels):
                if label != "":
                    # axes.annotate(label, (curve.jds[i], curve.mags[i]))
                    if len(label) > 4:  # Probably jd labels, we write vertically :
                        axes.annotate(label, (tmpjds[i], tmpmags[i]), xytext=(-3, -70), textcoords='offset points',
                                      size=12, color=curve.plotcolour, rotation=90)
                    else:  # horizontal writing
                        axes.annotate(label, (tmpjds[i], tmpmags[i]), xytext=(7, -6), textcoords='offset points',
                                      size=12, color=curve.plotcolour)

    if collapseref and len(reflevels) != 0:
        print("WARNING : collapsing the refs %s" % reflevels)
        if not hidecollapseref:
            axes.axhline(np.mean(np.array(reflevels)), color="gray", dashes=(3, 3))  # the new ref

    # The supplementary objects
    if len(splist) != 0:
        for stuff in splist:

            # We do some stupid type checking. But I like this as it does not require
            # to import spline and gpr etc.
            if hasattr(stuff, "knottype"):  # Then it's a spline
                spline = stuff
                if verbose:
                    print("#   %s -> %s" % (str(spline), str(spline.plotcolour)))

                npts = (spline.datapoints.jds[-1] - spline.datapoints.jds[0]) * 2.0
                xs = np.linspace(spline.datapoints.jds[0], spline.datapoints.jds[-1], int(npts))
                ys = spline.eval(jds=xs)
                axes.plot(xs, ys, "-", color=spline.plotcolour, zorder=+20, label=str(spline))
                # For the knots, we might not want to show them (by default we do show them) :
                if getattr(spline, "showknots", True):
                    if ax is not None:
                        ax.errorbar(spline.getinttex(), spline.eval(jds=spline.getinttex()),
                                    0.015 * np.ones(len(spline.getinttex())), capsize=0, ecolor=spline.plotcolour,
                                    linestyle="none", marker="", elinewidth=1.5, zorder=40, barsabove=True)
                        knotxs = spline.getinttex()
                        knotys = spline.eval(knotxs)
                        for (knotx, knoty) in zip(knotxs, knotys):
                            l = matplotlib.lines.Line2D([knotx, knotx], [knoty - knotsize, knoty + knotsize],
                                                        zorder=30,
                                                        linewidth=1.5, color=spline.plotcolour)
                            ax.add_line(l)
                    else:
                        axes = plt.gca()
                        knotxs = spline.getinttex()
                        knotys = spline.eval(knotxs)
                        for (knotx, knoty) in zip(knotxs, knotys):
                            l = matplotlib.lines.Line2D([knotx, knotx], [knoty - knotsize, knoty + knotsize],
                                                        zorder=30,
                                                        linewidth=1.5, color=spline.plotcolour)
                            axes.add_line(l)

                if showspldp:  # The datapoints of the spline (usually not shown)
                    axes.plot(spline.datapoints.jds, spline.datapoints.mags, marker=",", linestyle="None",
                              color=spline.plotcolour, zorder=-20)

            if hasattr(stuff, "pd"):  # Then it's a rslc
                rs = stuff
                axes.plot(rs.getjds(), rs.mags, "-", color=rs.plotcolour)
                xf = np.concatenate((rs.getjds(), rs.getjds()[::-1]))
                yf = np.concatenate((rs.mags + rs.magerrs, (rs.mags - rs.magerrs)[::-1]))
                axes.fill(xf, yf, facecolor=rs.plotcolour, alpha=0.2, edgecolor=(1, 1, 1), label=str(rs))

    # Astronomers like minor tick marks :
    minorxlocator = MultipleLocator(jdmintickstep)
    axes.xaxis.set_minor_locator(minorxlocator)

    if style:
        if "largeticks" in style:
            majorxlocator = MultipleLocator(jdtickstep)
            axes.xaxis.set_major_locator(majorxlocator)

    # Something for astronomers only : we invert the y axis direction !
    axes.set_ylim(axes.get_ylim()[::-1])

    if colourprop is not None and hidecolourbar is False:
        cbar = plt.colorbar(orientation='vertical', shrink=1.0, fraction=0.065, pad=0.025)
        cbar.set_label(colournicename)

    # And we make custom title :
    from matplotlib import rc
    rc('text', usetex=True)
    if title == "None" or title is None or title == "none":
        pass
    else:
        if titlexpos is None:
            axes.annotate(title, xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, -4),
                          textcoords='offset points', ha='center', va='top', fontsize=25)
        else:
            axes.annotate(title, xy=(titlexpos, 1.0), xycoords='axes fraction', xytext=(0, -4),
                          textcoords='offset points', ha='center', va='top', fontsize=25)
    rc('text', usetex=False)
    if jdrange is not None:
        axes.set_xlim(jdrange[0], jdrange[1])

    axes.set_xlabel("HJD - 2400000.5 (day)", fontsize=labelfontsize)
    if show_ylabel:
        axes.set_ylabel("Magnitude (relative)", fontsize=labelfontsize)

    if showdelays:
        txt = getnicetimedelays(lclist, separator="\n")
        axes.annotate(txt, xy=(0.77, 0.77), xycoords='axes fraction', xytext=(6, -6),
                      textcoords='offset points', ha='left', va='top')
        legendloc = 1
        if verbose:
            print("Delays between plotted curves :")
            print(txt)

    if showlegend and (len(lclist) > 0 or len(splist) > 0):
        axes.legend(loc=legendloc, numpoints=1, prop=fm.FontProperties(size=12))

    if magrange is not None:
        if type(magrange) == float or type(magrange) == int:
            # We find the mean mag of the stuff to plot :
            allmags = []
            for l in lclist:
                allmags.extend(l.getmags())
            meanlevel = np.mean(np.array(allmags))
            axes.set_ylim(meanlevel + magrange, meanlevel - magrange)
        else:
            axes.set_ylim(magrange[0], magrange[1])

    if showdates:  # Be careful when you change something here, it could mess up the axes.
        # Especially watch out when you change the plot range.
        # This showdates stuff should come at the very end
        minjd = axes.get_xlim()[0]
        maxjd = axes.get_xlim()[1]
        # axes.set_xlim(minjd, maxjd)
        yearx = axes.twiny()
        yearxmin = datetimefromjd(minjd + 2400000.5)
        yearxmax = datetimefromjd(maxjd + 2400000.5)
        yearx.set_xlim(yearxmin, yearxmax)
        yearx.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
        yearx.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        yearx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        yearx.xaxis.tick_top()
        if keeponlygrid:
            yearx.set_xticklabels([])

    minorylocator = MultipleLocator(magmintickstep)
    axes.yaxis.set_minor_locator(minorylocator)

    if showgrid:
        if showdates:
            yearx.grid(zorder=20, linestyle='dotted')
        else:
            axes.grid(zorder=20, linestyle='dotted')
        axes.yaxis.grid(False)

    if text is not None:
        for line in text:
            axes.text(line[0], line[1], line[2], transform=axes.transAxes, **line[3])

    if showinsert:
        assert insertname is not None
        from matplotlib.pyplot import imread
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        im = imread(insertname)
        imagebox = OffsetImage(im, zoom=0.5, interpolation="sinc", resample=True)
        ab = AnnotationBbox(imagebox, xy=(1.0, 1.0), xycoords='axes fraction', xybox=(-75, -75),
                            boxcoords="offset points",
                            pad=0.0, frameon=False
                            )
        axes.add_artist(ab)

    if showlogo:

        # The EPFL logo :
        from matplotlib.pyplot import imread
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        logodir = os.path.dirname(__file__)
        im = imread(os.path.join(logodir, "EPFL_Logo.png"))
        imagebox = OffsetImage(im, zoom=0.13, interpolation="sinc", resample=True)

        if logopos == "left":
            ab = AnnotationBbox(imagebox, xy=(0.0, 0.0), xycoords='axes pixels', xybox=(52, 30),
                                boxcoords="offset points",
                                pad=0.0, frameon=False
                                )
            axes.add_artist(ab)
            axes.annotate("COSMOGRAIL.org", xy=(0.0, 0.0), xycoords='axes fraction', fontsize=16, xytext=(105, 7),
                          textcoords='offset points', ha='left', va='bottom', color="gray")

        if logopos == "right":
            ab = AnnotationBbox(imagebox, xy=(1.0, 0.0), xycoords='axes fraction', xybox=(-200, 30),
                                boxcoords="offset points",
                                pad=0.0, frameon=False
                                )
            axes.add_artist(ab)
            axes.annotate("COSMOGRAIL.org", xy=(1.0, 0.0), xycoords='axes fraction', fontsize=16, xytext=(-10, 7),
                          textcoords='offset points', ha='right', va='bottom', color="gray")

        if logopos == "center":
            ab = AnnotationBbox(imagebox, xy=(0.55, 0.0), xycoords='axes fraction', xybox=(-80, 30),
                                boxcoords="offset points",
                                pad=0.0, frameon=False
                                )
            axes.add_artist(ab)
            axes.annotate("COSMOGRAIL.org", xy=(0.55, 0.0), xycoords='axes fraction', fontsize=16, xytext=(40, 7),
                          textcoords='offset points', ha='center', va='bottom', color="gray")

    if ax is not None:
        return

    if filename == "screen":
        plt.show()
    else:
        plt.savefig(filename, transparent=transparent)
        print("Plot written to %s" % filename)
        plt.close()  # this seems important so that the plot is not displayed when a next plt.show() is called.


def displayrange(lcs, margin=0.05):
    """
    returns a plausible range of mags and hjds to plot, so that you can keep this fixed in your plots
    :param lcs: LightCurve object
    :param margin: float
    """
    mags = []
    jds = []
    for l in lcs:
        mags.extend(l.getmags())
        jds.extend(l.getjds())

    magrange = np.max(mags) - np.min(mags)
    jdrange = np.max(jds) - np.min(jds)
    return ((np.min(jds) - margin * jdrange, np.max(jds) + margin * jdrange),
            (np.max(mags) + margin * magrange, np.min(mags) - margin * magrange))


def getnicetimedelays(lcs, separator="\n", to_be_sorted=False):
    """
    Returns a formatted piece of text of the time delays between a list of lc objects.

    :param lcs:list of LightCurve object
    :param separator: try ``" | "`` to get all the delays in one line.
    :type separator: string

    :param to_be_sorted: If True, I will sort my output according to l.object.
        But of course I will **not** modify the order of your lcs !
        By default this is False : if the curves are B, A, C and D in this order,
        I would return BA, BC, BD, AC, AD, CD

    :type to_be_sorted: boolean

    .. warning:: This is the function that **defines** the concept of "delay" (and its sign) used by
        pycs. **The delay AB corresponds to timeshift(B) - timeshift(A)**
        Other places where this is hard-coded :
        :py:func:`pycs.sim.plot.hists`

    """
    n = len(lcs)
    if to_be_sorted:
        worklcs = objsort(lcs, ret=True, verbose=False)
    else:
        worklcs = lcs
    couples = [(worklcs[i], worklcs[j]) for i in range(n) for j in range(n) if i < j]
    return separator.join(
        ["%s%s = %+7.2f" % (lc1.object, lc2.object, lc2.timeshift - lc1.timeshift) for (lc1, lc2) in couples])

def getdelays(lcs, to_be_sorted=False):
    """
    Return a list containing the relative time delay between curve
    :param lcs: list of LightCurve object
    :param to_be_sorted: If True, I will sort my output according to l.object.
        But of course I will **not** modify the order of your lcs !
        By default this is False : if the curves are B, A, C and D in this order,
        I would return BA, BC, BD, AC, AD, CD

    :type to_be_sorted: boolean

    :return: list of time delays.
    """
    n = len(lcs)
    if to_be_sorted:
        worklcs = objsort(lcs, ret=True, verbose=False)
    else:
        worklcs = lcs
    couples = [(worklcs[i], worklcs[j]) for i in range(n) for j in range(n) if i < j]
    return [lc2.timeshift - lc1.timeshift for (lc1, lc2) in couples]

def getnicetimeshifts(lcs, separator="\n"):
    """
    I return the timeshifts as a text string, use mainly for checks and debugging.
    """
    return separator.join(["%s  %+7.2f" % (l.object, l.timeshift) for l in lcs])


def gettimeshifts(lcs, includefirst=True):
    """
    I simply return the **absolute** timeshifts of the input curves as a numpy array.
    This is used by time shift optimizers, and usually not by the user.

    :param lcs:
    :param includefirst: If False, I skip the first of the shifts.
        For timeshift optimizers, it's indeed often ok not to move the first curve.
    :type includefirst: boolean

    """
    if includefirst:
        return np.array([l.timeshift for l in lcs])
    if not includefirst:
        return np.array([l.timeshift for l in lcs[1:]])


def settimeshifts(lcs, shifts, includefirst=False):
    """
    I set the timeshifts like those returned from :py:func:`gettimeshifts`.
    An do a little check.
    """

    if includefirst:
        assert len(lcs) == len(shifts)
        for (l, shift) in zip(lcs, shifts):
            l.timeshift = shift

    elif not includefirst:
        # print lcs, shifts
        assert len(lcs) - 1 == len(shifts)
        for (l, shift) in zip(lcs[1:], shifts):
            l.timeshift = shift


def shuffle(lcs):
    """
    I scramble the order of the objects in your list, in place.
    """
    random.shuffle(lcs)  # That was easier than I thought it would be when starting to wright this function ...


def objsort(lcs, ret=False, verbose=True):
    """
    I sort the lightcurve objects in your list (in place) according to their object names.

    :param lcs:
    :param verbose:
    :param ret: If True, I do not sort them in place, but return a new sorted list (containing the same objects).
    :type ret: boolean

    """

    # Maybe we start with some checks :
    diff_objects = set([l.object for l in lcs])
    if len(diff_objects) != len(lcs):
        raise RuntimeError("Cannot sort these objects : %s" % ", ".join([l.object for l in lcs]))

    # The actual sorting ...
    if not ret:
        lcs.sort(key=operator.attrgetter('object'))
        if verbose:
            print("Sorted lcs, order : %s" % ", ".join([l.object for l in lcs]))
        return
    if ret:
        return sorted(lcs, key=operator.attrgetter('object'))
