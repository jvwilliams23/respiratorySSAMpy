'''
Visualise error in morphological parameters diameter, length and branch angle
'''
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from math import ceil, floor
from sys import exit
import statsmodels.api as sm
from datetime import date
from distutils.util import strtobool
import argparse

date = str(date.today())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dirname', '-d',
                    default='morphologicalAnalysis/', 
                    type=str,#, required=True,
                    help='/path/to/results/\nshould contain *.txt files'
                    )
parser.add_argument('--blandaltman', '-ba',
                    default='False', 
                    type=strtobool,#, required=True,
                    help='Show bland altman plot in mm'
                    )
parser.add_argument('--normalised_error', '-err',
                    default='False', 
                    type=strtobool,#, required=True,
                    help='Show error vs value in percent'
                    )
parser.add_argument('--linear_regression', '-lr',
                    default='False', 
                    type=strtobool,#, required=True,
                    help='Show linear regression of ground truth v reconstruction'
                    )
parser.add_argument('--quiet', '-q',
                    default='True', 
                    type=strtobool,#, required=True,
                    help='Deliver quietly (no prints in terminal)'
                    )
parser.add_argument('--cutoff_level', '-c',
                    default=2, 
                    type=int,#, required=True,
                    help='do not show any generations above this value'
                    )
args = parser.parse_args()
quiet = args.quiet

# jw_cmap = ["#FF0000", "#00A08A", "blue", "#F98400", "#5BBCD6"]
jw_cmap = ['black', 'blue', 'green', 'red', "magenta", 
            "#00A08A", "pink", "#F98400", "#5BBCD6", 'salmon', 
            'yellow', 'orange']
marker_list = ['o', 'v', 's', 'X', 'd', '.', '^']

image_dir = 'images/reconstruction/morphologicalFigs/{}-'.format(date)

# parameters = ['diameter', 'length', 'angle']
# units = ['mm', 'mm', 'degrees']
parameters = ['diameter', 'lateralDistance', 'angle']
titles = ['Diameter', 'Maximum lateral distance', 'Angle']
units = ['mm', 'mm', 'degrees']

sampleIDs = glob("{}/angle*.txt".format(args.dirname))
param_file_name = args.dirname+"{}Stats{}.txt"
sampleIDs = [f.split('.txt')[0][-4:] for f in sampleIDs]
sampleIDs.sort()

def mscatter(x,y,ax=None, m=None, **kw):
  import matplotlib.markers as mmarkers
  if not ax: ax=plt.gca()
  sc = ax.scatter(x,y,**kw)
  if (m is not None) and (len(m)==len(x)):
      paths = []
      for marker in m:
          if isinstance(marker, mmarkers.MarkerStyle):
              marker_obj = marker
          else:
              marker_obj = mmarkers.MarkerStyle(marker)
          path = marker_obj.get_path().transformed(
                      marker_obj.get_transform())
          paths.append(path)
      sc.set_paths(paths)
  return sc

def bland_altman_plot(ax, data1, data2, marker='o', *args, **kwargs):
  data1     = np.asarray(data1)
  data2     = np.asarray(data2)
  mean      = np.mean([data1, data2], axis=0)
  diff      = data1 - data2                   # Difference between data1 and data2
  md        = np.mean(diff)                   # Mean of the difference
  sd        = np.std(diff, axis=0)            # Standard deviation of the difference

  if type(marker) == str:
    ax.scatter(mean, diff, *args, **kwargs)
  else:
    mscatter(mean, diff, ax=ax, m=marker, **kwargs)
  ax.axhline(md,           color='gray', linestyle='-')
  ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
  ax.axhline(md - 1.96*sd, color='gray', linestyle='--')

def ceilNearestN(x, base=5):
    return base * ceil(x/base)

def floorNearestN(x, base=5):
    return base * floor(x/base)

def bland_altman():
  global sampleIDs, parameters, units, marker_list, titles
  errors = [None]*len(parameters)
  # fig, ax = plt.subplots(1,3,figsize=(12,4))
  fig, ax = plt.subplots(1,len(parameters), figsize=(12,4))
  for i, (param, unit) in enumerate(zip(parameters, units)):
    gt_val = []
    reconstructed_val = []
    error_list = []
    colour_id = [] # create colour for each patient
    marker_id = [] # marker list for each patient
    for j, sample in enumerate(sampleIDs):
      error_file_data = np.loadtxt(param_file_name.format(param, sample),
                                    skiprows=1)
      # # drop trachea from length calcs as true length is unknown
      # if param == 'length':
      #   error_file_data = error_file_data[1:]
      #   label_start = 1 # offset marker list so consistent across plots
      #   if not quiet: print(error_file_data)
      # else:
      #   label_start = 0
      # format columns of text file into arrays
      gen_level = error_file_data[:,0] # airway bifurcation level
      # do not show points from generation below a certain depth 
      delete_rows = np.where(gen_level>args.cutoff_level)[0]
      error_file_data = np.delete(error_file_data, delete_rows, axis=0)
      # get all column data from text file
      gt_val.append(error_file_data[:,1]) # ground truth value for param
      reconstructed_val.append(error_file_data[:,2]) # reconstructed value for param
      abs_error = error_file_data[:,-2] # absolute error 
      pct_error = error_file_data[:,-1] # relative error in percent
      error_list.append(pct_error)
      sample_col = jw_cmap[j] # colour for this sample
      col_list = [sample_col]*len(reconstructed_val[-1]) # for all points in sample  
      # print(len(reconstructed_val[-1]))
      # exit()
      if not quiet: print(param)
      if not quiet: print('branch, gen, shape')
      # create list of markers to represent different bifurcation levels
      sample_marker_list = []
      for s in range(0, len(reconstructed_val[-1])):
        if not quiet: print(s, int(gen_level[s]), marker_list[int(gen_level[s])])
        sample_marker_list.extend( marker_list[int(gen_level[s])] )
      colour_id.extend(col_list)
      marker_id.extend(sample_marker_list)
    errors[i] = np.array(error_list)
    plot_values = np.c_[np.hstack(reconstructed_val), np.hstack(gt_val)]
    # plot_values = np.c_[np.hstack(gt_val), np.hstack(reconstructed_val)]

    # correlation_line = np.linspace(np.min(plot_values), 
    #                                 np.max(plot_values), 100)
    bland_altman_plot(ax[i], plot_values[:,0], plot_values[:,1], 
                      c=colour_id, marker=marker_id)
    # sm.graphics.mean_diff_plot(plot_values[:,0], plot_values[:,1], ax=ax[i], 
    #                             scatter_kwds=scatter_props)
    # sns.set(font_scale=0.1)
    ax[i].set_ylabel('Difference [{}]'.format(unit), fontsize=12)
    ax[i].set_xlabel('Means [{}]'.format(unit), fontsize=12)
    ax[i].set_title("{}".format(titles[i]), fontsize=14)
    # if param == 'length':
    #   ax[i].set_xscale('log')
  import matplotlib.lines as mlines
  legend = [mlines.Line2D([], [], color='black', marker=marker_list[0], ls='None',
                           label='level 0 (trachea)'),
            mlines.Line2D([], [], color='black', marker=marker_list[1], ls='None',
                           label='level 1 (main bronchi)'),
            mlines.Line2D([], [], color='black', marker=marker_list[2], ls='None',
                           label='level 2')][:args.cutoff_level+1]
  for j, sample in enumerate(sampleIDs):
    sampleMarker = [mlines.Line2D([], [], ls='None',
                                  color=jw_cmap[j], marker=marker_list[0], 
                                  label='sample {}'.format(sample))]
    legend.extend(sampleMarker)
  plt.legend(handles=legend, fontsize=11, frameon=False, 
              bbox_to_anchor=(1.5,0.5), loc='center')

  plt.subplots_adjust(top=0.9,
  bottom=0.125,
  left=0.07,
  right=0.8,
  hspace=0.2,
  wspace=0.355)
  plt.show()
  # exit()

def linear_regression():
  global parameters, sampleIDs, units, titles
  errors = [None]*len(parameters)
  plt.close()
  # fig, ax = plt.subplots(1,3,figsize=(12,4))
  fig, ax = plt.subplots(1,len(parameters), figsize=(12,4))
  for i, param in enumerate(parameters):
    gt_val = []
    reconstructed_val = []
    error_list = []
    colour_id = [] # create colour for each patient
    for j, sample in enumerate(sampleIDs):
      error_file_data = np.loadtxt(param_file_name.format(param, sample),
                                    skiprows=1)
      gen_level = error_file_data[:,0] # airway bifurcation level
      # do not show points from generation below a certain depth 
      delete_rows = np.where(gen_level>args.cutoff_level)[0]
      error_file_data = np.delete(error_file_data, delete_rows, axis=0)

      gt_val.append(error_file_data[:,1])
      reconstructed_val.append(error_file_data[:,2])
      abs_error = error_file_data[:,-2]
      pct_error = error_file_data[:,-1]
      error_list.append(pct_error)
      # col_list = [jw_cmap[j]]*len(reconstructed_val[-1]) 
      # colour_id.extend(col_list)
    errors[i] = np.array(error_list)
    plot_values = np.c_[np.hstack(gt_val), np.hstack(reconstructed_val)]

    # ax[i].boxplot(errors[i])
    correlation_line = np.linspace(np.min(plot_values), 
                                    np.max(plot_values), 100)
    ax[i].plot(correlation_line, correlation_line, alpha=0.2,c='black')
    ax[i].scatter(plot_values[:,0], plot_values[:,1],
                  #c=colour_id
                  )

    ticks = ax[i].get_yticks()
    if ticks.size > 8:
      ticks = ticks[::2]
    ax[i].set_xticks(ticks)
    ax[i].set_yticks(ticks)
    ax[i].set_title(titles[i], fontsize=12)
    if param == 'length' and args.cutoff_level >= 2:
      ax[i].set_xscale('log')
      ax[i].set_yscale('log')
    ax[i].set_ylabel('reconstruction [{}]'.format(units[i]), fontsize=11)
    ax[i].set_xlabel('ground truth [{}]'.format(units[i]), fontsize=11)
    # ax[i].axis('equal')
  plt.subplots_adjust(top=0.905, bottom=0.125,
                      left=0.09, right=0.95, 
                      hspace=0.2, wspace=0.275)
  plt.show()
  # plt.savefig(image_dir+'correlation_subplots.pdf')

  # exit()

def error_gt():
  global parameters, sampleIDs, units, titles
  errors = [None]*len(parameters)
  # fig, ax = plt.subplots(1,3,figsize=(12,4))
  fig, ax = plt.subplots(1,len(parameters), figsize=(12,4))
  for i, param in enumerate(parameters):
    gt_val = []
    reconstructed_val = []
    error_list = []
    colour_id = []
    marker_id = []
    for j, sample in enumerate(sampleIDs):
      error_file_data = np.loadtxt(param_file_name.format(param, sample),
                                    skiprows=1)
      gen_level = error_file_data[:,0] # airway bifurcation level
      delete_rows = np.where(gen_level>args.cutoff_level)[0]
      error_file_data = np.delete(error_file_data, delete_rows, axis=0)
      gt_val.append(error_file_data[:,1])
      reconstructed_val.append(error_file_data[:,2])
      abs_error = error_file_data[:,-2]
      pct_error = error_file_data[:,-1]
      error_list.append(pct_error)
      if not quiet: print(len(jw_cmap))
      col_list = [jw_cmap[j]]*len(reconstructed_val[-1]) 
      # colour_id.extend(col_list)
      # create list of markers to represent different bifurcation levels
      sample_marker_list = []
      for s in range(0, len(reconstructed_val[-1])):
        if not quiet: print(s, int(gen_level[s]), marker_list[int(gen_level[s])])
        sample_marker_list.extend( marker_list[int(gen_level[s])] )
      colour_id.extend(col_list)
      marker_id.extend(sample_marker_list)
    errors[i] = np.array(error_list)
    plot_values = np.c_[np.hstack(gt_val), np.hstack(error_list)]
    ax[i].axhline(y=0, alpha=0.2, c='black')
    if not quiet: print(colour_id)
    # ax[i].scatter(plot_values[:,0], plot_values[:,1],
    #               c=colour_id)
    mscatter(plot_values[:,0], plot_values[:,1], ax=ax[i],
             c=colour_id, m=marker_id)
    ax[i].set_title(titles[i], fontsize=12)
    if param == 'length' and args.cutoff_level >= 2:
      ax[i].set_xscale('log')
    # ax[i].set_ylabel('error [{}]'.format(units[i]), fontsize=11)
    ax[i].set_ylabel('error [{}]'.format('%'), fontsize=11)
    ax[i].set_xlabel('ground truth [{}]'.format(units[i]), fontsize=11)
    # ax[i].axis('equal')

  # create legend
  import matplotlib.lines as mlines
  legend = [mlines.Line2D([], [], color='black', marker=marker_list[0], ls='None',
                           label='level 0 (trachea)'),
            mlines.Line2D([], [], color='black', marker=marker_list[1], ls='None',
                           label='level 1 (main bronchi)'),
            mlines.Line2D([], [], color='black', marker=marker_list[2], ls='None',
                           label='level 2')][:args.cutoff_level+1]

  for j, sample in enumerate(sampleIDs):
    sampleMarker = [mlines.Line2D([], [], ls='None',
                                  color=jw_cmap[j], marker=marker_list[0], 
                                  label='sample {}'.format(sample))]
    legend.extend(sampleMarker)
  plt.legend(handles=legend, fontsize=11, frameon=False, 
              bbox_to_anchor=(1.5,0.5), loc='center')
  plt.subplots_adjust(top=0.905, bottom=0.125,
                      left=0.06, right=0.8,
                      hspace=0.225, wspace=0.295)
  # plt.savefig(image_dir+'error_with_gtval.pdf')
  plt.show()

if args.blandaltman:
  bland_altman()
if args.linear_regression:
  linear_regression()
if args.normalised_error:
  error_gt()
