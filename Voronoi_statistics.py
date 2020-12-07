#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, mad_std


def get_errors(data, used_cols = None):
   """
   This routine finds the corresponding error columns for the columns in "used_cols"
   """

   errors = pd.DataFrame(index = data.index)

   cols = data.columns
   if not used_cols:
      used_cols = cols

   for col in used_cols:
      if '%s_error'%col in cols:
         errors['%s_error'%col] = data['%s_error'%col]
      else:
         errors['%s_error'%col] = np.ones_like(data[col])*1e-32

   return errors


def median(table):
   """
   This routine calculates the median and the error of the median.
   The formula is 1.25*sigma/sqrt(N) for medians of normal distributions. This is the upper limmit of error.
   """

   var_err = (1.25404*table.std()/np.sqrt(table.count())).add_prefix('median_').add_suffix('_error')
   var_median = table.median().add_prefix('median_')

   return pd.concat([var_median, var_err])


def weighted_avg_err(table):
   """
   This routine calculates the error-weighted mean and its error.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   x_i = table.loc[:, var_cols]

   ex_i = get_errors(table, used_cols = var_cols)

   var_variance = 1./(1./ex_i**2).sum(axis = 0)

   var_avg = (x_i.div(ex_i.values**2)).sum(axis = 0) * var_variance.values

   return pd.concat([var_avg.add_prefix('mean_'), np.sqrt(var_variance[~var_variance.index.duplicated()]).add_prefix('mean_')])


def sigma_clip_cell(table, var_cols = None, sigma = 3):
   """
   This routine performs sigma clipping of the data whithin each Vornoi Cell
   """

   filtered_data = sigma_clip(table.loc[:, var_cols], sigma=sigma, maxiters=None, stdfunc=mad_std)

   index_filt = table[~filtered_data.mask.any(axis = 1)].index
   
   return index_filt


def voronoi_stats(table, coo, name_vor_bin, n_stars = 50, sigma = None):
   """
   This routine calls voronoi_binning using as coordinates those in "coo" and
   compute the mean and the median values for all columns in the table within these cells.
   """
   
   if len(coo) == 3:
      from voronoi_3d_binning import voronoi_3d_binning as voronoi_binning
   else:
      from voronoi_2d_binning import voronoi_2d_binning as voronoi_binning

   var_cols = [x for x in table.columns if not '_error' in x]

   if sigma is not None:
      table_it = table.loc[:, coo+list(set(var_cols) - set(coo))].copy()
      convergence = False
      while not convergence:

         vor_bin, vor_n_stars = voronoi_binning(table_it.loc[:,coo].values, n_stars, quiet =1, wvt = True, cvt = True)

         table_it[name_vor_bin] = vor_bin

         clean_indexes = table_it.groupby([name_vor_bin]).apply(sigma_clip_cell, var_cols = list(set(var_cols) - set(coo)), sigma = sigma)
         clean_index = list(set([index for clean_index in clean_indexes for index in clean_index]))
         
         table_clean = table_it.iloc[table_it.index.isin(clean_index), :]

         convergence = table_it.equals(table_clean)

         table_it = table_clean.copy()

      sigma_clip_used = table_it.index
      sigma_clip_rejected = table.index[~table.index.isin(table_it.index)]
      
      del [table_it, table_clean]

      table = table[table.index.isin(sigma_clip_used)].copy()
   else:
      vor_bin, vor_n_stars = voronoi_binning(table.loc[:,coo].values, n_stars, quiet =1, wvt = True, cvt = True)

   table[name_vor_bin] = vor_bin

   grouped_table_mean = table.groupby([name_vor_bin]).apply(weighted_avg_err).reset_index()
   grouped_table_median = table.groupby([name_vor_bin])[var_cols].apply(median).reset_index()

   results = grouped_table_mean.drop(columns=['mean_%s'%name_vor_bin, 'mean_%s_error'%name_vor_bin, ]).merge(grouped_table_median, on = [name_vor_bin], how = 'left')
   results['vor_n_stars'] = vor_n_stars
   
   return table, results, vor_bin


def voronoi_stats_launcher(args):
   """
   This routine pipes voronoi_stats arguments into multiple threads.
   """

   return voronoi_stats(*args)


def voronoi_stats_parallel(table, voronoi_vars, voronoi_coos, voronoi_names, voronoi_nstars, sigma = None, use_parallel = False):
   """
   This routine launch the Voronoi tesellation
   """

   from multiprocessing import Pool, cpu_count

   args = []
   for voronoi_coo, voronoi_name, voronoi_nstar in zip(voronoi_coos, voronoi_names, voronoi_nstars):
      args.append((table.loc[:,voronoi_vars], voronoi_coo, voronoi_name, voronoi_nstar, sigma))

   if use_parallel:
      pool = Pool(min(cpu_count(), voronoi_coos))
      Y = pool.map(voronoi_stats_launcher, args)
   else:
      Y = []
      for arg in args:
         Y.append(voronoi_stats_launcher(arg))

   tables_filtered = [results[0] for results in Y]
   tables_voronoi = [results[1] for results in Y]
   cells_voronoi = [results[2] for results in Y]
   
   return tables_filtered, tables_voronoi, cells_voronoi


if __name__=="__main__":
   """
   This is an using example
   """
   
   # Generate some mock data
   n_points = 5000
   signal_error = 10.

   mean = [0, 0]

   cov = [[5.0, -1],
          [-1, 5.0]]

   coo = np.random.multivariate_normal(mean, cov, n_points)
   signal = coo[:,0]+coo[:,1]
   signal_with_error = signal + np.random.normal(0, signal_error, n_points)

   table = pd.DataFrame(data = {'x': coo[:,0], 'y': coo[:,1], 'z': signal_with_error, 'z_error': np.ones_like(signal)*signal_error})

   table_filt, voronoi_tables, voronoi_cells = voronoi_stats_parallel(table, ['x', 'y', 'z', 'z_error' ], [['x','y']], ['voronoi_cell'], [100], sigma = 3)
   
   # We select the first table
   table_filt = table_filt[0].merge(voronoi_tables[0], on = ['voronoi_cell'], how = 'left')

   fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharex = True, sharey = True, figsize = (10, 4))

   ax1.scatter(table.x, table.y, c = signal, alpha = 0.75, vmin = signal.min(), vmax = signal.max())
   ax1.annotate('Original', xy=(table.x.min(), table.y.min()), xycoords='data', color = 'k')

   ax2.scatter(table.x, table.y, c = signal_with_error)
   ax2.annotate('Noise added', xy=(table.x.min(), table.y.min()), xycoords='data', color = 'k')

   ax3.scatter(table_filt.x, table_filt.y, c = table_filt.mean_z, alpha = 0.75, vmin = signal.min(), vmax = signal.max())
   ax3.annotate('Recovered', xy=(table_filt.x.min(), table_filt.y.min()), xycoords='data', color = 'k')

   ax4.scatter(table_filt.x, table_filt.y, c = table_filt.voronoi_cell, alpha = 0.75, )
   ax4.annotate('Voronoi cells', xy=(table_filt.x.min(), table_filt.y.min()), xycoords='data', color = 'k')

   ax1.set_xlabel('x')
   ax1.set_ylabel('y')
   
   plt.show()

"""
Andres del Pino Molina
"""
