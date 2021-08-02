'''
@author: Josh Williams

Plotting functions for ssm

'''

import numpy as np
import matplotlib.pyplot as plt
from vtkplotter import *
from matplotlib.lines import Line2D
from math import ceil

colList = [(0.47843137254901963, 0.47843137254901963, 0.47843137254901963),
			(1.0,170.0/255.0,0.0), (1.0, 0.3333333333333333, 0.4980392156862745), 
			(0.0, 0.6666666666666666, 1.0),
			(100.0/255.0, 220/255.0, 184.0/255.0)]

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plotReconErr(error, tag=""):
	'''
		Plots reconstruction error.
		This is the model's ability to reconstruct shapes in the training set

		Input: 2D array of errors. 1D is landmark error, 
				other is for each training iteration
	'''
	if tag != "":
		tag = "-" + tag

	ntrain = error.shape[0]
	plt.close()
	fig, ax = plt.subplots()
	xlim = ntrain+1
	xi = np.arange(0,ntrain)+1#np.arange(1,ntrain)
	y = np.mean(error,axis=1)
	yerr = np.std(error,axis=1)/(ntrain-1)

	# plt.ylim(min(y)*0.9,1.01)
	plt.xlim(1, xlim)
	plt.plot(xi, y, marker='o',ms=4, linestyle='-', color='b')
	plt.errorbar(xi, y, yerr=yerr, label='both limits (default)')

	plt.xlabel('Number of Components')
	plt.xticks(xi[::3]) 
	plt.ylabel('Error %')
	plt.title('Reconstruction error')

	ax.grid(axis='x')
	# plt.show()
	plt.savefig("images/reconstruction-error-chart"+tag+".pdf")

def plotGenErr(error, tag=""):
	'''
		Plots generalisation error.
		This is the model's ability to reconstruct shapes in the testing set

		Input: 2D array of errors. 1D is landmark error, 
				other is for each training iteration
	'''
	if tag != "":
		tag = "-" + tag
	ntrain = error.shape[0]
	plt.close()
	fig, ax = plt.subplots()
	xlim = ntrain+1
	xi = np.arange(0,ntrain)+1#np.arange(1,ntrain)
	y = np.mean(error,axis=1)
	yerr = np.std(error,axis=1)/np.sqrt(ntrain-1)

	# plt.ylim(min(y)*0.9,1.01)
	plt.xlim(1, xlim)
	plt.plot(xi, y, marker='o',ms=4, linestyle='-', color='b')
	plt.errorbar(xi, y, yerr=yerr, label='both limits (default)')

	plt.xlabel('Number of Components')
	plt.xticks(xi[::3]) 
	plt.ylabel('Error %')
	plt.title('Generalisation error')

	ax.grid(axis='x')
	# plt.show()
	plt.savefig("images/generalisation-error-chart"+tag+".pdf")

def plotSpecErr(error, tag=""):
	'''
		Plots specificity error.
		This is the model's ability to reconstruct random shapes

		Input: 2D array of errors. 1D is landmark error, 
				other is for each training iteration
	'''
	if tag != "":
		tag = "-" + tag
	ntrain = error.shape[0]
	plt.close()
	fig, ax = plt.subplots()
	xlim = ntrain+1
	xi = np.arange(0,ntrain)+1#np.arange(1,ntrain)
	y = error[:,0]
	yerr = error[:,1]

	# plt.ylim(min(y)*0.9,1.01)
	plt.xlim(1, xlim)
	plt.plot(xi, y, marker='o',ms=4, linestyle='-', color='b')
	plt.errorbar(xi, y, yerr=yerr, label='both limits (default)')

	plt.xlabel('Number of Components')
	plt.xticks(xi[::3]) 
	plt.ylabel('Error %')
	plt.title('Specificity error')

	ax.grid(axis='x')
	# plt.show()
	plt.savefig("images/specificity-error-chart"+tag+".pdf")

def myround(x, base=5):
	return base * ceil(x/base)

def plotSSMmetrics(compac, recon, gen, spec, tag="", shapeName='shape1'):
	if tag != "":
		tag = "-" + tag
	plt.close()
	fig, ax = plt.subplots(2,2, figsize=cm2inch(17,17))#, sharex=True)
	colList = ["black"]
	col = "black"
	labels = [shapeName]

	ntrain = compac.shape[0]
	xi = np.arange(0,ntrain)+1
	print(xi, xi.shape)
	y = np.mean(compac, axis=1)
	yerr = np.std(compac, axis=1)
	ax[0][0].plot(xi, y, marker='o',ms=2, linestyle='-', 
				color=col, mec=col, mfc=col)
	ax[0][0].set_title('Compactness', fontsize=12)
	ax[0][0].errorbar(xi, y, yerr=yerr,c=col, label='both limits (default)')
	ax[0][0].set_ylabel('Variance [%]', fontsize=11)
	ax[0][0].set_xticks(np.arange(0,myround(xi.shape[0], 5)+1,5))
	ax[0][0].set_xlim(0, myround(xi.shape[0], 5)+1)
	ax[0][0].grid(axis='x')
	

	ntrain = recon.shape[0]
	xi = np.arange(0,ntrain)+1
	y = np.mean(recon,axis=1)
	yerr = np.std(recon,axis=1)/np.sqrt(ntrain-1)
	ax[0][1].plot(xi, y, marker='o',ms=2, linestyle='-', 
				color=col, mec=col, mfc=col)
	ax[0][1].set_title('Reconstruction error', fontsize=12)
	ax[0][1].errorbar(xi, y, yerr=yerr,c=col, label='both limits (default)')
	ax[0][1].set_ylabel('Error [mm]', fontsize=11)
	ax[0][1].set_xticks(np.arange(0,myround(xi.shape[0], 5)+1,5))
	ax[0][1].set_xlim(0, myround(xi.shape[0], 5)+1)
	ax[0][1].grid(axis='x')
	
	y = np.mean(gen,axis=1)
	yerr = np.std(gen,axis=1)/np.sqrt(ntrain-1)
	ax[1][0].plot(xi, y, marker='o',ms=2, linestyle='-', 
				color=col, mec=col, mfc=col)
	ax[1][0].set_title('Generalisation error', fontsize=12)
	ax[1][0].errorbar(xi, y, yerr=yerr,c=col, label='both limits (default)')
	ax[1][0].set_ylabel('Error [mm]', fontsize=11)
	ax[1][0].set_xlabel('Components', fontsize=11)
	ax[1][0].set_xticks(np.arange(0,myround(xi.shape[0], 5)+1,5))
	ax[1][0].set_xlim(0, myround(xi.shape[0], 5)+1)
	ax[1][0].grid(axis='x')

	y = spec[:,0]
	yerr = spec[:,1]
	ax[1][1].plot(xi, y, marker='o',ms=2, linestyle='-', 
				color=col, mec=col, mfc=col)
	ax[1][1].set_title('Specificity error', fontsize=12)
	ax[1][1].errorbar(xi, y, yerr=yerr,c=col, label='both limits (default)')
	ax[1][1].set_ylabel('Error [mm]', fontsize=11)
	ax[1][1].set_xlabel('Components', fontsize=11)
	ax[1][1].set_xticks(np.arange(0,myround(xi.shape[0], 5)+1,5))
	ax[1][1].set_xlim(0, myround(xi.shape[0], 5)+1)
	ax[1][1].grid(axis='x')

	lines = [Line2D([0],[0], marker='o',ms=2, linestyle='-', 
						color=col, mec=col, mfc=col) for col in colList]
	plt.legend(lines, labels,
				bbox_to_anchor=(-0.2, -0.4), loc='lower center', 
				ncol=5,#len(compac.keys()), 
				 fontsize=11,\
				 frameon=False)#loc=[0.0, 0.7]
	plt.subplots_adjust(hspace=0.35, wspace=0.3, 
						top=0.95, bottom=0.15, right=0.95) 

	plt.savefig("images/SSM/metrics-error-chart"+tag+".pdf")
	return None