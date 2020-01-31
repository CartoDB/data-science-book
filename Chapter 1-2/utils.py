import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import shapely

def Variogram_plot(v, fig_title=None, axes=None, grid=True, show=False, hist=True):
        """Variogram Plot
        Plot the experimental variogram, the fitted theoretical function and
        an histogram for the lag classes. The axes attribute can be used to
        pass a list of AxesSubplots or a single instance to the plot
        function. Then these Subplots will be used. If only a single instance
        is passed, the hist attribute will be ignored as only the variogram
        will be plotted anyway.
        Parameters
        ----------
        axes : list, tuple, array, AxesSubplot or None
            If None, the plot function will create a new matplotlib figure.
            Otherwise a single instance or a list of AxesSubplots can be
            passed to be used. If a single instance is passed, the hist
            attribute will be ignored.
        grid : bool
            Defaults to True. If True a custom grid will be drawn through
            the lag class centers
        show : bool
            Defaults to True. If True, the show method of the passed or
            created matplotlib Figure will be called before returning the
            Figure. This should be set to False, when used in a Notebook,
            as a returned Figure object will be plotted anyway.
        hist : bool
            Defaults to True. If False, the creation of a histogram for the
            lag classes will be suppressed.
        Returns
        -------
        matplotlib.Figure
        """
        # get the parameters
        _bins = v.bins
        _exp = v.experimental
        x = np.linspace(0, np.nanmax(_bins), 100)  # make the 100 a param?

        # do the plotting
        if axes is None:
            if hist:
                fig = plt.figure(figsize=(8, 5))
                ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
                ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)
                fig.subplots_adjust(hspace=0)
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
                ax2 = None
        elif isinstance(axes, (list, tuple, np.ndarray)):
            ax1, ax2 = axes
            fig = ax1.get_figure()
        else:
            ax1 = axes
            ax2 = None
            fig = ax1.get_figure()

        # apply the model
        y = v.transform(x)

        # handle the relative experimental variogram
        if v.normalized:
            _bins /= np.nanmax(_bins)
            y /= np.max(_exp)
            _exp /= np.nanmax(_exp)
            x /= np.nanmax(x)

        # ------------------------
        # plot Variograms
        ax1.plot(_bins, _exp, marker=".", color='orange', markersize=15, linestyle='None')
        ax1.plot(x, y, 'blue', linewidth=2)
        ax1.set_facecolor('white')

        # ax limits
        if v.normalized:
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])
        if grid:
            ax1.grid('off')
            ax1.vlines(_bins, *ax1.axes.get_ybound(), colors=(.85, .85, .85),
                       linestyles='dashed',linewidth=0.5)
        # annotation
        ax1.axes.set_ylabel('semivariance (%s)' % v._estimator.__name__)
        ax1.axes.set_xlabel('Lag (-)')

        # ------------------------
        # plot histogram
        if ax2 is not None and hist:
            # calc the histogram
            _count = np.fromiter(
                (g.size for g in v.lag_classes()), dtype=int
            )

            # set the sum of hist bar widths to 70% of the x-axis space
            w = (np.max(_bins) * 0.7) / len(_count)

            # plot
            ax2.bar(_bins, _count, width=w, align='center', color='blue')

            # adjust
            plt.setp(ax2.axes.get_xticklabels(), visible=False)
            ax2.axes.set_yticks(ax2.axes.get_yticks()[1:])

            # need a grid?
            if grid:
                ax2.grid('off')
                ax2.vlines(_bins, *ax2.axes.get_ybound(),
                           colors=(.85, .85, .85), linestyles='dashed',linewidth=0.5)

            # anotate
            ax2.axes.set_ylabel('N')
            ax2.set_facecolor('white')

        plt.title(fig_title)
        return fig

def geom2gdf(geom, crs, lonlat = True):
    geom = [['geom', geom]]
    geom = pd.DataFrame(geom, columns = ['geom', 'geometry'])
    geom = gpd.GeoDataFrame(geom, geometry=geom.geometry)
    geom.crs = {'init': crs}
    if(lonlat):
        geom = geom.to_crs({'init': 'epsg:4326'})

    return geom

def ell2gdf(M, sMx, sMy, theta, crs):
    circ = shapely.geometry.Point(M).buffer(1)
    ell  = shapely.affinity.scale(circ, int(sMx), int(sMy))
    ellr = shapely.affinity.rotate(ell,-np.degrees(theta))
    poly = geom2gdf(ellr, crs)

    return poly

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.
    Parameters
    ----------
    nc_fid : netCDF4.Dataset
       A netCDF4 dateset object
    verb : Boolean
       whether or not nc_attrs, nc_dims, and nc_vars are printed
    Returns
    -------
    nc_attrs : list
       A Python list of the NetCDF file global attributes
    nc_dims : list
       A Python list of the NetCDF file dimensions
    nc_vars : list
       A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key
        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)
    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars