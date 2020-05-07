'''
Copyright (c) 2009-2013, Adam Ginsburg

All rights reserved.

Redistribution and use in source and binary forms, with or without modification
,are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name Adam Ginsburg nor the names of other contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Code adapted from Adam Ginsburg's gaussfitter routines
https://github.com/keflavich/gaussfitter
'''

import numpy as np
from typing import List

__all__ = ["moments", "twodgaussian"]


def moments(data: np.ndarray, angle_guess=90.0) -> List[float]:
    """
    Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.

    Parameters
    ----------

    data : np.ndarray
        data from which the Gaussian parameters will be calculated

    angle_guess : float, optional
        Guess of the angle of the Gaussian, defual 5 degrees

    Returns
    -------
    params : List[floats]
        List of parameters of the Gaussian.

    """

    y = int(data.shape[1] / 2)
    x = int(data.shape[0] / 2)

    col = data[int(y), :]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum() / np.abs(col).sum())

    row = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum() / np.abs(row).sum())

    height = np.median(data.ravel())
    amplitude = data.max() - height

    params = [amplitude, x, y]

    if np.isnan((width_y, width_x, height, amplitude)).any():
        raise ValueError("something is nan")

    params = [height] + params
    params = params + [width_x, width_y]

    params = params + [angle_guess*np.pi/180.]

    return params


def twodgaussian(xydata, offset: float, amplitude: float, xo: float, yo: float,
                 sigma_x: float, sigma_y: float, theta: float):
    """
    Returns a 2d gaussian function of the form:
    x' = np.cos(rota) * x - np.sin(rota) * y
    y' = np.sin(rota) * x + np.cos(rota) * y
    (rota should be in degrees)
    g = b + a * np.exp ( - ( ((x-center_x)/width_x)**2 +
    ((y-center_y)/width_y)**2 ) / 2 )

    inpars = [b,a,center_x,center_y,width_x,width_y,rota]
             (b is background height, a is peak amplitude)

    where x and y are the input parameters of the returned function,
    and all other parameters are specified by this function


    Parameters
    ----------

    xydata : List[float], List[float]
        Stack of x and y values values. xydata[0] is x and xydata[1] is y
    offset : float
        Offset or height of the Gaussian distribution.
    amplitude: float
        Amplitude of Gaussian.
    xo, yo: float
        Centre point of Gaussian distribution.
    sigma_x, sigma_y : float
        Standard deviation of Gaussian distribution in x and y directions.
    theta : float
        Angle of Gaussian distribution.

    Returns
    -------
    g.ravel() : List[float]
        1D array of computed Gaussian distribution. Array is 1D so that
        function is compatible with Scpiy's curve_fit.
        Parameters are: Height/offset, amplitude, xo, yo, sigx, sigy, theta

    """

    height = offset
    center_y, center_x = xo, yo
    width_x, width_y = sigma_x, sigma_y

    rota = theta
    rcen_x = center_x * np.cos(rota) - center_y * np.sin(rota)
    rcen_y = center_x * np.sin(rota) + center_y * np.cos(rota)

    x = xydata[0].reshape((xydata[0].shape[0], xydata[0].shape[1]))
    y = xydata[1].reshape((xydata[1].shape[0], xydata[1].shape[1]))

    xp = x * np.cos(rota) - y * np.sin(rota)
    yp = x * np.sin(rota) + y * np.cos(rota)

    g = height + amplitude * np.exp(-(((rcen_x-xp)/width_x)**2 +
                                    ((rcen_y-yp)/width_y)**2)/2.)

    return g.ravel()
