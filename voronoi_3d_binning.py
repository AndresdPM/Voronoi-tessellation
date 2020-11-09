"""
#####################################################################

Copyright (C) 2001-2016, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your
research, we would appreciate an acknowledgment to use of
`the Voronoi binning method by Cappellari & Copin (2003)'.

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

#####################################################################

NAME:
    VORONOI_2D_BINNING

AUTHOR:
      Michele Cappellari, University of Oxford
      michele.cappellari_at_physics.ox.ac.uk

PURPOSE:
      Perform adaptive spatial binning of Integral-Field Spectroscopic
      (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
      This method is required for the proper analysis of IFS
      observations, but can also be used for standard photometric
      imagery or any other two-dimensional data.
      This program precisely implements the algorithm described in
      section 5.1 of the reference below.

EXPLANATION:
      Further information on VORONOI_2D_BINNING algorithm can be found in
      Cappellari M., Copin Y., 2003, MNRAS, 342, 345
      http://adsabs.harvard.edu/abs/2003MNRAS.342..345C

CALLING SEQUENCE:

        binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetSN,
                               cvt=True, pixelsize=None, plot=True,
                               quiet=True, sn_func=None, wvt=True)

INPUTS:
           X: Vector containing the X coordinate of the pixels to bin.
              Arbitrary units can be used (e.g. arcsec or pixels).
              In what follows the term "pixel" refers to a given
              spatial element of the dataset (sometimes called "spaxel" in
              the IFS community): it can be an actual pixel of a CCD
              image, or a spectrum position along the slit of a long-slit
              spectrograph or in the field of view of an IFS
              (e.g. a lenslet or a fiber).
              It is assumed here that pixels are arranged in a regular
              grid, so that the pixel size is a well defined quantity.
              The pixel grid however can contain holes (some pixels can be
              excluded from the binning) and can have an irregular boundary.
              See the above reference for an example and details.
           Y: Vector (same size as X) containing the Y coordinate
              of the pixels to bin.
      SIGNAL: Vector (same size as X) containing the signal
              associated with each pixel, having coordinates (X,Y).
              If the `pixels' are actually the apertures of an
              integral-field spectrograph, then the signal can be
              defined as the average flux in the spectral range under
              study, for each aperture.
              If pixels are the actual pixels of the CCD in a galaxy
              image, the signal will be simply the counts in each pixel.
       NOISE: Vector (same size as X) containing the corresponding
              noise (1 sigma error) associated with each pixel.
    TARGETSN: The desired signal-to-noise ratio in the final
              2D-binned data. E.g. a S/N~50 per pixel may be a
              reasonable value to extract stellar kinematics
              information from galaxy spectra.

KEYWORDS:
         CVT: Set this keyword to skip the Centroidal Voronoi Tessellation
              (CVT) step (vii) of the algorithm in Section 5.1 of
              Cappellari & Copin (2003).
              This may be useful if the noise is strongly non Poissonian,
              the pixels are not optimally weighted, and the CVT step
              appears to introduces significant gradients in the S/N.
              A similar alternative consists of using the /WVT keyword below.
     SN_FUNC: Generic function to calculate the S/N of a bin with spaxels
              "index" with the form: "sn = func(index, signal, noise)".
              If this keyword is not set, or is set to None, the program
              uses the _sn_func(), included in the program file, but
              another function can be adopted if needed.
              See the documentation of _sn_func() for more details.
       QUIET: by default the program shows the progress while accreting
              pixels and then while iterating the CVT. Set this keyword
              to avoid printing progress results.
         WVT: When this keyword is set, the routine bin2d_cvt_equal_mass is
              modified as proposed by Diehl & Statler (2006, MNRAS, 368, 497).
              In this case the final step of the algorithm, after the bin-accretion
              stage, is not a modified Centroidal Voronoi Tessellation, but it uses
              a Weighted Voronoi Tessellation.
              This may be useful if the noise is strongly non Poissonian,
              the pixels are not optimally weighted, and the CVT step
              appears to introduces significant gradients in the S/N.
              A similar alternative consists of using the /NO_CVT keyword above.
              If you use the /WVT keyword you should also include a reference to
              `the WVT modification proposed by Diehl & Statler (2006).'

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage

#----------------------------------------------------------------------------

def _sn_func(index, signal=None):
    """
    Default function to calculate the S/N of a bin with spaxels "index".

    The Voronoi binning algorithm does not require this function to have a
    specific form and this default one can be changed by the user if needed
    by passing a different function as

        ... = voronoi_2d_binning(..., sn_func=sn_func)

    The S/N returned by sn_func() does not need to be an analytic
    function of S and N.

    There is also no need for sn_func() to return the actual S/N.
    Instead sn_func() could return any quantity the user needs to equalize.

    For example sn_func() could be a procedure which uses ppxf to measure
    the velocity dispersion from the coadded spectrum of spaxels "index"
    and returns the relative error in the dispersion.

    Of course an analytic approximation of S/N, like the one below,
    speeds up the calculation.

    :param index: integer vector of length N containing the indices of
        the spaxels for which the combined S/N has to be returned.
        The indices refer to elements of the vectors signal and noise.
    :param signal: vector of length M>N with the signal of all spaxels.
    :param noise: vector of length M>N with the noise of all spaxels.
    :return: scalar S/N or another quantity that needs to be equalized.
    """

    #sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))
    
    sn = np.sum(signal[index])

    # The following commented line illustrates, as an example, how one
    # would include the effect of spatial covariance using the empirical
    # Eq.(1) from http://adsabs.harvard.edu/abs/2015A%26A...576A.135G
    # Note however that the formula is not accurate for large bins.
    #
    # sn /= 1 + 1.07*np.log10(index.size)

    return  sn

#----------------------------------------------------------------------

def _weighted_centroid(x, y, z, density):
    """
    Computes weighted centroid of one bin.
    Equation (4) of Cappellari & Copin (2003)

    """
    mass = np.sum(density)
    xBar = np.sum(x*density)/mass
    yBar = np.sum(y*density)/mass
    zBar = np.sum(z*density)/mass

    return xBar, yBar, zBar

#----------------------------------------------------------------------

def _accretion(x, y, z, signal, targetSN, quiet, sn_func):
    """
    Implements steps (i)-(v) in section 5.1 of Cappellari & Copin (2003)

    """
    n = x.size
    classe = np.zeros(n, dtype=int)  # will contain the bin number of each given pixel
    good = np.zeros(n, dtype=bool)   # will contain 1 if the bin has been accepted as good

    currentBin = np.argmax(signal)  # Start from the pixel with highest S/N
    SN = signal[currentBin]

    # Rough estimate of the expected final bin number.
    # This value is only used to give an idea of the expected
    # remaining computation time when binning very big dataset.
    #
    w = signal < targetSN
    maxnum = int(np.sum(signal[w]**2)/targetSN**2 + np.sum(~w))

    # The first bin will be assigned CLASS = 1
    # With N pixels there will be at most N bins
    #
    for ind in range(1, n+1):

        if not quiet:
            print(ind, ' / ', maxnum)

        classe[currentBin] = ind  # Here currentBin is still made of one pixel
        
        xBar, yBar, zBar = x[currentBin], y[currentBin], z[currentBin]    # Centroid of one pixels

        while True:

            if np.all(classe):
                break  # Stops if all pixels are binned

            # Find the unbinned pixel closest to the centroid of the current bin
            #
            unBinned = np.flatnonzero(classe == 0)
            k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2 + (z[unBinned] - zBar)**2)

            # (1) Find the distance from the closest pixel to the current bin

            # (2) Estimate the `roundness' of the POSSIBLE new bin
            nextBin = np.append(currentBin, unBinned[k])
            #roundness = _roundness(x[nextBin], y[nextBin], pixelSize)

            # (3) Compute the S/N one would obtain by adding
            # the CANDIDATE pixel to the current bin
            #
            SNOld = SN
            SN = sn_func(nextBin, signal)

            # Test whether (1) the CANDIDATE pixel is connected to the
            # current bin, (2) whether the POSSIBLE new bin is round enough
            # and (3) whether the resulting S/N would get closer to targetSN
            #
            if (abs(SN - targetSN) > abs(SNOld - targetSN) or SNOld > SN):
                if SNOld > 0.8*targetSN:
                   good[currentBin] = 1
                break

            # If all the above 3 tests are negative then accept the CANDIDATE
            # pixel, add it to the current bin, and continue accreting pixels
            #
            classe[unBinned[k]] = ind
            currentBin = nextBin

            # Update the centroid of the current bin
            #
            xBar, yBar, zBar = np.mean(x[currentBin]), np.mean(y[currentBin]), np.mean(z[currentBin])

        # Get the centroid of all the binned pixels
        #
        binned = classe > 0
        if np.all(binned):
            break  # Stop if all pixels are binned
        xBar, yBar, zBar = np.mean(x[binned]), np.mean(y[binned]), np.mean(z[binned])

        # Find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel.
        #
        unBinned = np.flatnonzero(classe == 0)
        k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2 + (z[unBinned] - zBar)**2)
        currentBin = unBinned[k]    # The bin is initially made of one pixel
        SN = signal[currentBin]

    classe *= good  # Set to zero all bins that did not reach the target S/N

    return classe

#----------------------------------------------------------------------------

def _reassign_bad_bins(classe, x, y, z):
    """
    Implements steps (vi)-(vii) in section 5.1 of Cappellari & Copin (2003)

    """
    # Find the centroid of all successful bins.
    # CLASS = 0 are unbinned pixels which are excluded.
    #
    good = np.unique(classe[classe > 0])
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)
    znode = ndimage.mean(z, labels=classe, index=good)

    # Reassign pixels of bins with S/N < targetSN
    # to the closest centroid of a good bin
    #
    bad = classe == 0
    index = np.argmin((x[bad, None] - xnode)**2 + (y[bad, None] - ynode)**2 + (z[bad, None] - ynode)**2, axis=1)
    classe[bad] = good[index]

    # Recompute all centroids of the reassigned bins.
    # These will be used as starting points for the CVT.
    #
    good = np.unique(classe)
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)
    znode = ndimage.mean(z, labels=classe, index=good)

    return xnode, ynode, znode

#----------------------------------------------------------------------------

def _cvt_equal_mass(x, y, z, signal, xnode, ynode, znode, quiet, wvt, sn_func):
    """
    Implements the modified Lloyd algorithm
    in section 4.1 of Cappellari & Copin (2003).

    NB: When the keyword WVT is set this routine includes
    the modification proposed by Diehl & Statler (2006).

    """
    if wvt:
        dens = np.ones_like(signal)
    else:
        dens = signal**2  # See beginning of section 4.1 of CC03
    scale = np.ones_like(xnode)   # Start with the same scale length for all bins

    for it in range(1, xnode.size):  # Do at most xnode.size iterations

        xnodeOld, ynodeOld, znodeOld = xnode.copy(), ynode.copy(), znode.copy()

        # Computes (Weighted) Voronoi Tessellation of the pixels grid
        #
        if x.size < 10000:
            classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2 + (z[:, None] - znode)**2)/scale**2, axis=1)
        else:  # use for loop to reduce memory usage
            classe = np.zeros(x.size, dtype=int)
            for j, (xj, yj, zj) in enumerate(zip(x, y, z)):
                classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2 + (zj - znode)**2)/scale**2)

        # Computes centroids of the bins, weighted by dens**2.
        # Exponent 2 on the density produces equal-mass Voronoi bins.
        # The geometric centroids are computed if WVT keyword is set.
        #
        good = np.unique(classe)
        for k in good:
            index = np.flatnonzero(classe == k)   # Find subscripts of pixels in bin k.
            xnode[k], ynode[k], znode[k] = _weighted_centroid(x[index], y[index], z[index], dens[index]**2)
            if wvt:
                sn = sn_func(index, signal)
                scale[k] = np.sqrt(index.size/sn)  # Eq. (4) of Diehl & Statler (2006)

        diff = np.sum((xnode - xnodeOld)**2 + (ynode - ynodeOld)**2 + (znode - znodeOld)**2)

        if not quiet:
            print('Iter: %4i  Diff: %.4g' % (it, diff))

        if diff == 0:
            break

    # If coordinates have changed, re-compute (Weighted) Voronoi Tessellation of the pixels grid
    #
    if diff > 0:
        if x.size < 100000:
            classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2 + (z[:, None] - znode)**2)/scale**2, axis=1)
        else:  # use for loop to reduce memory usage
            classe = np.zeros(x.size, dtype=int)
            for j, (xj, yj, zj) in enumerate(zip(x, y, z)):
                classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2 + (zj - znode)**2)/scale**2)
        good = np.unique(classe)  # Check for zero-size Voronoi bins

    # Only return the generators and scales of the nonzero Voronoi bins

    return xnode[good], ynode[good], znode[good], scale[good], it

#-----------------------------------------------------------------------

def _compute_useful_bin_quantities(x, y, z, signal, xnode, ynode, znode, scale, sn_func):
    """
    Recomputes (Weighted) Voronoi Tessellation of the pixels grid to make sure
    that the class number corresponds to the proper Voronoi generator.
    This is done to take into account possible zero-size Voronoi bins
    in output from the previous CVT (or WVT).

    """
    # classe will contain the bin number of each given pixel
    #
    if x.size < 100000:
        classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2 + (z[:, None] - znode)**2)/scale**2, axis=1)
    else:  # use for loop to reduce memory usage
        classe = np.zeros(x.size, dtype=int)
        for j, (xj, yj, zj) in enumerate(zip(x, y, z)):
            classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2 + (zj - znode)**2)/scale**2)

    # At the end of the computation evaluate the bin luminosity-weighted
    # centroids (xbar, ybar) and the corresponding final S/N of each bin.
    #
    xbar = np.empty_like(xnode)
    ybar = np.empty_like(xnode)
    zbar = np.empty_like(xnode)
    
    sn = np.empty_like(xnode)
    volume = np.empty_like(xnode)
    good = np.unique(classe)
    for k in good:
        index = np.flatnonzero(classe == k)   # index of pixels in bin k.
        xbar[k], ybar[k], zbar[k] = _weighted_centroid(x[index], y[index], z[index], signal[index])
        sn[k] = sn_func(index, signal)
        volume[k] = index.size

    return classe, xbar, ybar, zbar, sn, volume

#----------------------------------------------------------------------

def voronoi_3d_binning(data, targetSN, cvt=True,
                         quiet=True, sn_func=None, wvt=True):

    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    signal = np.ones_like(x)

    if not (x.size == y.size == z.size == signal.size):
        raise ValueError('Input vectors (x, y, signal) must have the same size')

    if sn_func is None:
        sn_func = _sn_func

    if np.min(signal) > targetSN:
        raise ValueError('All pixels have enough S/N and binning is not needed')

    print('Bin-accretion...')
    classe = _accretion(x, y, z, signal, targetSN, quiet, sn_func)
    xNode, yNode, zNode = _reassign_bad_bins(classe, x, y, z)
    if cvt:
        xNode, yNode, zNode, scale, it = _cvt_equal_mass(x, y, z, signal, xNode, yNode, zNode, quiet, wvt, sn_func)
    else:
        scale = 1.
    classe, xBar, yBar, zBar, sn, volume = _compute_useful_bin_quantities(x, y, z, signal, xNode, yNode, zNode, scale, sn_func)
    w = volume == 1

    return classe, sn

#----------------------------------------------------------------------------
