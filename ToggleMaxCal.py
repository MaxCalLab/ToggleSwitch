#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:23:10 2017

@author: tefirman
"""

""" Conditions of Simulation and Inference """

simConditions = 'ToggleGillespie_RNA_7days'   # Name of the simulation conditions
timeInc = 300                   # Sampling time (delta T)
numDays = 7                     # Total time length of each simulation
numTrials = 100                 # Number of trajectories in each simulation
maxN = 59                       # Max possible number of proteins (for FSP purposes) (old value = 48)
maxn = 10                       # Max possible number of mRNA (for FSP purposes)
numIterations = 200             # Number of frames to project forward for the likelihood (old value = 390)
simNum = 10                     # Index of which simulation to infer from
minMaxl_a = 25                  # Minimum value of m to calculate likelihood for
maxMaxl_a = 40                  # Maximum value of m to calculate likelihood for

""" Index of CUDA Enabled GPU Device """
gpuDevice = 0

""" Filename of simulation to look for/create  """
fname = simConditions + '_' + str(simNum) + '.npz'

""" Filename of parameter file to create/save """
fout = 'ExtractedParameters_ToggleSwitch_MaxCal_'+ str(simConditions) + '_Sim' + str(simNum) + '_SameMaxN_' + str(numIterations) + 'iterations.npz'

""" Python Modules Necessary for Function """

import pyopencl
import pyopencl.array
import pyopencl.clmath
from pyopencl.elementwise import ElementwiseKernel
import os
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize
import datetime
import cupy as cp

""" Initializing Context for PyOpenCL and CUPY """
cp.cuda.Device(gpuDevice).use()
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
os.environ['PYOPENCL_NO_CACHE'] = '1'
platform = pyopencl.get_platforms()
my_gpu_devices = [platform[0].get_devices(device_type=pyopencl.device_type.GPU)[gpuDevice]]

ctx = pyopencl.Context(devices=my_gpu_devices)
queue = pyopencl.CommandQueue(ctx)
mf = pyopencl.mem_flags

def conditionsInitGill(g,g_pro,g_rep,g_prorep,d,p,r,k_f,k_b,exclusive,\
n_A_init,n_a_init,n_alpha_init,n_B_init,n_b_init,n_beta_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all of your stated conditions """
    return {'g':g, 'g_pro':g_pro, 'g_rep':g_rep, 'g_prorep':g_prorep, \
            'd':d, 'p':p, 'r':r, 'k_f':k_f, 'k_b':k_b, 'exclusive':exclusive, \
            'N_A_init':n_A_init, 'N_a_init':n_a_init, 'N_alpha_init':n_alpha_init, \
            'N_B_init':n_B_init, 'N_b_init':n_b_init, 'N_beta_init':n_beta_init, \
            'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def gillespieSim(conditions,n_A,n_a,n_alpha,n_B,n_b,n_beta):
    """ Gillespie simulation used to generate the synthetic toggle switch input data """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_a = (conditions['N_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_alpha = (conditions['N_alpha_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_B = (conditions['N_B_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_b = (conditions['N_b_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_beta = (conditions['N_beta_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_a) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            numa = np.copy(conditions['N_a_init'])
            numalpha = np.copy(conditions['N_alpha_init'])
            numB = np.copy(conditions['N_B_init'])
            numb = np.copy(conditions['N_b_init'])
            numbeta = np.copy(conditions['N_beta_init'])
            n_A.append([np.copy(numA)])
            n_a.append([np.copy(numa)])
            n_alpha.append([np.copy(numalpha)])
            n_B.append([np.copy(numB)])
            n_b.append([np.copy(numb)])
            n_beta.append([np.copy(numbeta)])
        else:
            numA = np.copy(n_A[numTrial][-1])
            numa = np.copy(n_a[numTrial][-1])
            numalpha = np.copy(n_alpha[numTrial][-1])
            numB = np.copy(n_B[numTrial][-1])
            numb = np.copy(n_b[numTrial][-1])
            numbeta = np.copy(n_beta[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            if conditions['exclusive']:
                prob = [conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numalpha*(1 - numbeta),\
                conditions['g_rep']*numbeta*(1 - numalpha),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numa,\
                conditions['p']*numa,\
                conditions['r']*numA,\
                conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numbeta*(1 - numalpha),\
                conditions['g_rep']*numalpha*(1 - numbeta),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numb,\
                conditions['p']*numb,\
                conditions['r']*numB,\
                conditions['k_f']*(1 - numalpha - numbeta)*numA,\
                conditions['k_b']*numalpha,\
                conditions['k_f']*(1 - numalpha - numbeta)*numB,\
                conditions['k_b']*numbeta]
            else:
                prob = [conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numalpha*(1 - numbeta),\
                conditions['g_rep']*numbeta*(1 - numalpha),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numa,\
                conditions['p']*numa,\
                conditions['r']*numA,\
                conditions['g']*(1 - numalpha)*(1 - numbeta),\
                conditions['g_pro']*numbeta*(1 - numalpha),\
                conditions['g_rep']*numalpha*(1 - numbeta),\
                conditions['g_prorep']*numalpha*numbeta,\
                conditions['d']*numb,\
                conditions['p']*numb,\
                conditions['r']*numB,\
                conditions['k_f']*(1 - numalpha)*numA,\
                conditions['k_b']*numalpha,\
                conditions['k_f']*(1 - numbeta)*numB,\
                conditions['k_b']*numbeta]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA).tolist())
                n_a[numTrial].append(np.copy(numa).tolist())
                n_alpha[numTrial].append(np.copy(numalpha).tolist())
                n_B[numTrial].append(np.copy(numB).tolist())
                n_b[numTrial].append(np.copy(numb).tolist())
                n_beta[numTrial].append(np.copy(numbeta).tolist())
                incCheckpoint += conditions['inc']
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 <= sum(prob[:4]):
                numa += 1
            elif randNum2 <= sum(prob[:5]):
                numa -= 1
            elif randNum2 <= sum(prob[:6]):
                numA += 1
            elif randNum2 <= sum(prob[:7]):
                numA -= 1
            elif randNum2 <= sum(prob[:11]):
                numb += 1
            elif randNum2 <= sum(prob[:12]):
                numb -= 1
            elif randNum2 <= sum(prob[:13]):
                numB += 1
            elif randNum2 <= sum(prob[:14]):
                numB -= 1
            elif randNum2 <= sum(prob[:15]):
                numalpha += 1
                numA -= 1
            elif randNum2 <= sum(prob[:16]):
                numalpha -= 1
                numA += 1
            elif randNum2 <= sum(prob[:17]):
                numbeta += 1
                numB -= 1
            else:
                numbeta -= 1
                numB += 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
        n_a[numTrial] = n_a[numTrial][:conditions['numSteps'] + 1]
        n_alpha[numTrial] = n_alpha[numTrial][:conditions['numSteps'] + 1]
        n_B[numTrial] = n_B[numTrial][:conditions['numSteps'] + 1]
        n_b[numTrial] = n_b[numTrial][:conditions['numSteps'] + 1]
        n_beta[numTrial] = n_beta[numTrial][:conditions['numSteps'] + 1]
    return n_A, n_a, n_alpha, n_B, n_b, n_beta

def peakVals(origHist,numFilter,minVal):
    """ Robustly identifies peak locations for 2D probability distributions """
    simHist = np.copy(origHist)
    for numTry in range(numFilter):
        for ind1 in range(len(simHist)):
            for ind2 in range(len(simHist[ind1])):
                simHist[ind1,ind2] = np.sum(simHist[max(ind1 - 1,0):min(ind1 + 2,len(simHist)),\
                max(ind2 - 1,0):min(ind2 + 2,len(simHist[ind1]))])/\
                np.size(simHist[max(ind1 - 1,0):min(ind1 + 2,len(simHist)),\
                max(ind2 - 1,0):min(ind2 + 2,len(simHist[ind1]))])
    maxInds = []
    for ind1 in range(len(simHist)):
        for ind2 in range(len(simHist[ind1])):
            if simHist[ind1,ind2] == np.max(simHist[max(ind1 - 1,0):min(ind1 + 2,len(simHist)),\
            max(ind2 - 1,0):min(ind2 + 2,len(simHist[ind1]))]) and simHist[ind1,ind2] >= minVal:
                maxInds.append([ind1,ind2])
    return maxInds

def entropyStats(n_A,n_B,maxInds):
    """ Calculates the different entropy values for a given set of trajectories """
    global maxN
    stateProbs = [np.zeros((maxN**2,maxN**2)) for ind in range(len(maxInds))]
    cgProbs = np.zeros((len(maxInds),len(maxInds)))
    dwellVals = [[] for ind in range(len(maxInds))]
    for numTrial in range(len(n_A)):
        tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        cgTraj = -1*np.ones(len(n_A[numTrial]))
        cgTraj[np.all([n_A[numTrial] <= maxInds[0][0],n_B[numTrial] >= maxInds[0][1]],axis=0)] = 0
        cgTraj[np.all([n_A[numTrial] >= maxInds[1][0],n_B[numTrial] <= maxInds[1][1]],axis=0)] = 1
        ind1 = np.where(cgTraj >= 0)[0][0]
        inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        while len(inds) > 0:
            stateProbs[int(cgTraj[ind1])] += np.histogram2d(maxN*n_B[numTrial][ind1 + 1:ind1 + inds[0]] + \
            n_A[numTrial][ind1 + 1:ind1 + inds[0]],maxN*n_B[numTrial][ind1:ind1 + inds[0] - 1] + \
            n_A[numTrial][ind1:ind1 + inds[0] - 1],bins=np.arange(-0.5,maxN**2))[0]
            cgProbs[int(cgTraj[ind1 + inds[0]]),int(cgTraj[ind1])] += 1
            cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += inds[0]
            dwellVals[int(cgTraj[ind1])].append(inds[0])
            ind1 += inds[0]
            inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        stateProbs[int(cgTraj[ind1])] += np.histogram2d(maxN*n_B[numTrial][ind1 + 1:] + \
        n_A[numTrial][ind1 + 1:],maxN*n_B[numTrial][ind1:-1] + n_A[numTrial][ind1:-1],\
        bins=np.arange(-0.5,maxN**2))[0]
        cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += len(n_A[numTrial]) - ind1 - 1
    totProbs = np.zeros((maxN**2,maxN**2))
    stateEntropies = []
    for ind in range(len(stateProbs)):
        totProbs += stateProbs[ind]
        stateProbs[ind] = stateProbs[ind]/np.sum(stateProbs[ind])
        stateEntropies.append(-np.nansum(stateProbs[ind]*np.log2(stateProbs[ind])))
    totProbs = totProbs/np.sum(totProbs)
    totEntropy = -np.nansum(totProbs*np.log2(totProbs))
    cgProbs = cgProbs/np.sum(cgProbs)
    macroEntropy = -np.nansum(cgProbs*np.log2(cgProbs))
    return totEntropy, stateEntropies, macroEntropy, dwellVals

def conditionsInitMaxCal(h_a,h_A,k_Aa,k_Ab,maxl_a,n_A_init,l_a_init,l_iA_init,\
n_B_init,l_b_init,l_iB_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all of your stated conditions """
    return {'h_a':float(h_a), 'h_A':float(h_A), 'k_Aa':float(k_Aa), 'k_Ab':float(k_Ab), \
    'maxl_a':maxl_a, 'N_A_init':n_A_init, 'l_a_init':l_a_init, 'l_iA_init':l_iA_init, \
    'N_B_init':n_B_init, 'l_b_init':l_b_init, 'l_iB_init':l_iB_init, \
    'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def logFactorial(value):
    """ Returns the logarithm of the factorial of the value provided using Sterling's approximation """
    if all([value > 0,abs(round(value) - value) < 0.000001,value <= 34]):
        return float(sum(np.log(range(1,int(value) + 1))))
    elif all([value > 0,abs(round(value) - value) < 0.000001,value > 34]):
        return float(value)*np.log(float(value)) - float(value) + \
        0.5*np.log(2.0*np.pi*float(value)) - 1.0/(12.0*float(value))
    elif value == 0:
        return float(0)
    else:
        return float('nan')

""" Calculating and storing all possible values of n-choose-k for faster fitting """
factMat = float('Inf')*np.ones((maxN,maxN + 1),dtype='float32')
for nVal in range(maxN):
    for kVal in range(nVal + 1):
        factMat[nVal][kVal] = logFactorial(nVal) - logFactorial(kVal) - logFactorial(nVal - kVal)
    del kVal
del nVal
factMat_GPU = pyopencl.array.to_device(queue, factMat)

"""
Defining the elementwise kernel for GPU calculation of transition probabilities
Translation of index to transition variables:
l_A = i/((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))
l_a = i%((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))/((n_B + 1)*(maxl_a + 1))
l_B = i%((n_B + 1)*(maxl_a + 1))/(maxl_a + 1)
l_b = i%(maxl_a + 1)
"""
prob_init = ElementwiseKernel(ctx,
        "float *x, float *factVals, float h_a, float h_A, float k_Aa, float k_Ab, int maxl_a, int n_A, int n_B, int maxN", \
        "x[i] = factVals[(maxN + 1)*n_A + i/((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))] + " + \
        "factVals[(maxN + 1)*n_B + i%((maxl_a + 1)*(n_B + 1))/(maxl_a + 1)] + " + \
        "h_a*(i%((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))/((n_B + 1)*(maxl_a + 1))) + " + \
        "h_A*(i/((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))) + " + \
        "k_Aa*(i%((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))/((n_B + 1)*(maxl_a + 1)))*(i/((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))) + " + \
        "k_Ab*(i%(maxl_a + 1))*(i/((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))) + " + \
        "h_a*(i%(maxl_a + 1)) + h_A*(i%((maxl_a + 1)*(n_B + 1))/(maxl_a + 1)) + " + \
        "k_Aa*(i%(maxl_a + 1))*(i%((maxl_a + 1)*(n_B + 1))/(maxl_a + 1)) + " + \
        "k_Ab*(i%((maxl_a + 1)*(n_B + 1)*(maxl_a + 1))/((n_B + 1)*(maxl_a + 1)))*(i%((maxl_a + 1)*(n_B + 1))/(maxl_a + 1))",
        "prob_init")

"""
Defining the kernel to enter transition matrix values via the GPU
Translation of index to protein numbers:
finalN_A = idx%maxN
finalN_B = idx/maxN
"""
prg = pyopencl.Program(ctx, """
  __kernel void fsp_add(__global float *fspMat, __global float *probMat, ushort n_A, ushort n_B, ushort maxl_a, ushort maxN)
  {
    int idx = get_global_id(0);
    if (idx < maxN*maxN){
        for (int l_a = 0; l_a <= maxl_a; l_a++){
            for (int l_b = 0; l_b <= maxl_a; l_b++){
                if (idx%maxN - l_a >= 0 && idx/maxN - l_b >= 0){
                    if (idx%maxN - l_a <= n_A && idx/maxN - l_b <= n_B){
                        fspMat[maxN*maxN*idx + maxN*n_B + n_A] += probMat[(idx%maxN - l_a)*(maxl_a + 1)*(n_B + 1)*(maxl_a + 1) + l_a*(n_B + 1)*(maxl_a + 1) + (idx/maxN - l_b)*(maxl_a + 1) + l_b];
                        }
                    }
                }
            }
        }
  }
  """).build()

def maxCalSim(conditions,n_A,l_a,l_iA,n_B,l_b,l_iB):
    """ Monte Carlo simulation used to generate trajectories with Maximum Caliber """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    global maxN
    global maxl_a
    global factMat_GPU
    probsTot = [[[] for ind in range(maxN)] for ind in range(maxN)]
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_a = (conditions['l_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_iA = (conditions['l_iA_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_B = (conditions['N_B_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_b = (conditions['l_b_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_iB = (conditions['l_iB_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        if len(n_A) < numTrial + 1:
            n_A.append(np.copy(conditions['N_A_init'][0]).tolist())
            l_a.append(np.copy(conditions['l_a_init'][0]).tolist())
            l_iA.append(np.copy(conditions['l_iA_init'][0]).tolist())
            n_B.append(np.copy(conditions['N_B_init'][0]).tolist())
            l_b.append(np.copy(conditions['l_b_init'][0]).tolist())
            l_iB.append(np.copy(conditions['l_iB_init'][0]).tolist())
        for numStep in range(len(n_A[numTrial]),len(n_A[numTrial]) + conditions['numSteps']):
            randNum = np.random.rand(1)
            if len(probsTot[n_A[numTrial][numStep - 1]][n_B[numTrial][numStep - 1]]) == 0:
                probsMC = pyopencl.array.empty(queue,(n_A[numTrial][numStep - 1] + 1,conditions['maxl_a'] + 1,\
                n_B[numTrial][numStep - 1] + 1,conditions['maxl_a'] + 1),dtype='float32')
                prob_init(probsMC,factMat_GPU,conditions['h_a'],conditions['h_A'],conditions['k_Aa'],conditions['k_Ab'],conditions['maxl_a'],\
                n_A[numTrial][numStep - 1],n_B[numTrial][numStep - 1],maxN)
                probsMC = pyopencl.clmath.exp(probsMC - pyopencl.array.max(probsMC).get())
                probsMC /= pyopencl.array.sum(probsMC).get()
                probsTot[n_A[numTrial][numStep - 1]][n_B[numTrial][numStep - 1]] = probsMC.get()
            probSum = 0
            l_aVal = -1
            while l_aVal < conditions['maxl_a'] and randNum > probSum:
                l_aVal += 1
                l_iAVal = n_A[numTrial][numStep - 1] + 1
                while l_iAVal > 0 and randNum > probSum:
                    l_iAVal -= 1
                    l_bVal = -1
                    while l_bVal < conditions['maxl_a'] and randNum > probSum:
                        l_bVal += 1
                        l_iBVal = n_B[numTrial][numStep - 1] + 1
                        while l_iBVal > 0 and randNum > probSum:
                            l_iBVal -= 1
                            probSum += probsTot[n_A[numTrial][numStep - 1]]\
                            [n_B[numTrial][numStep - 1]]\
                            [l_iAVal,l_aVal,l_iBVal,l_bVal]
            n_A[numTrial].append(l_iAVal + l_aVal)
            l_a[numTrial].append(l_aVal)
            l_iA[numTrial].append(l_iAVal)
            n_B[numTrial].append(l_iBVal + l_bVal)
            l_b[numTrial].append(l_bVal)
            l_iB[numTrial].append(l_iBVal)
    return n_A, l_a, l_iA, n_B, l_b, l_iB

def maxCalFSP(lagrangeVals,maxl_a,numSteps,n_A_init,n_B_init):
    """ Finite State Projection for Maximum Caliber """
    """ Essentially calculates the 2D probability distribution for some later point in time """
    global maxN
    global factMat_GPU
    global ctx
    global mf
    global timeVals
    probMatrix = np.zeros((maxN**2,maxN**2),dtype='float32')
    probMatrix_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, probMatrix.nbytes)
    for n_A in range(maxN):
        if (n_A%10) == 0:
            tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
            tempData.write('N_A = ' + str(n_A) + '\n')
            tempData.close()
        for n_B in range(maxN):
            probs_GPU = pyopencl.array.empty(queue,(n_A + 1,maxl_a + 1,n_B + 1,maxl_a + 1),dtype='float32')
            prob_init(probs_GPU,factMat_GPU,lagrangeVals[0],lagrangeVals[1],lagrangeVals[2],lagrangeVals[3],maxl_a,n_A,n_B,maxN)
            probs_GPU = pyopencl.clmath.exp(probs_GPU - pyopencl.array.max(probs_GPU).get())
            probs_GPU /= pyopencl.array.sum(probs_GPU).get()
            prg.fsp_add(queue,(probMatrix.shape[0],1),None,probMatrix_buf,probs_GPU.data,\
            np.uint16(n_A),np.uint16(n_B),np.uint16(maxl_a),np.uint16(maxN))
    probMatrix = np.empty_like(probMatrix)
    pyopencl.enqueue_copy(queue,probMatrix,probMatrix_buf)
    probMatrix = np.hstack((probMatrix,np.zeros((maxN**2,1))))
    probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
    probInit = np.zeros(len(probMatrix))
    probInit[maxN*n_B_init + n_A_init] = 1.0
    startTime = datetime.datetime.now()
    probFinal = np.dot(np.linalg.matrix_power(probMatrix,numSteps),probInit)[:-1].reshape((maxN,maxN))
    endTime = datetime.datetime.now()
    timeVals.append((endTime - startTime).total_seconds())
    return probFinal

def maxCalEquil(lagrangeVals,maxl_a):
    """ Finds eigenvectors of the transition matrix to calculate the """
    """ effective equilibrium distribution of protein number in Maximum Caliber. """
    """ lagrangeVals = ['h_a','h_A','k_Aa','k_Ab'] """
    global maxN
    global factMat_GPU
    global ctx
    global mf
    global timeVals
    probMatrix = np.zeros((maxN**2,maxN**2),dtype='float32')
    probMatrix_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, probMatrix.nbytes)
    for n_A in range(maxN):
        for n_B in range(maxN):
            probs_GPU = pyopencl.array.empty(queue,(n_A + 1,maxl_a + 1,n_B + 1,maxl_a + 1),dtype='float32')
            prob_init(probs_GPU,factMat_GPU,lagrangeVals[0],lagrangeVals[1],lagrangeVals[2],lagrangeVals[3],maxl_a,n_A,n_B,maxN)
            probs_GPU = pyopencl.clmath.exp(probs_GPU - pyopencl.array.max(probs_GPU).get())
            probs_GPU /= pyopencl.array.sum(probs_GPU).get()
            prg.fsp_add(queue,(probMatrix.shape[0],1),None,probMatrix_buf,probs_GPU.data,\
            np.uint16(n_A),np.uint16(n_B),np.uint16(maxl_a),np.uint16(maxN))
    probMatrix = np.empty_like(probMatrix)
    pyopencl.enqueue_copy(queue,probMatrix,probMatrix_buf)
    probMatrix = scipy.sparse.csc_matrix(probMatrix - np.identity(probMatrix.shape[0]))
    val,prob = scipy.sparse.linalg.eigs(probMatrix,k=1,sigma=0)
    prob = prob.reshape(len(prob))
    if any([np.imag(val) > 0.001,np.real(val) > 0.001,any(abs(np.imag(prob)) > 0.001),\
    not ((max(np.real(prob)) < 0.0001) ^ (min(np.real(prob)) > -0.0001))]):
        finalProb = np.array([])
        return finalProb
    prob = prob.astype('float64')
    prob /= sum(prob)
    return prob.reshape((maxN,maxN))

def dwellHistMaxCal(conditions,maxInds,maxTol,inc):
    """ Analytical method of calculating the dwell time probability distribution via FSP """
    global maxN
    dwellProbs = []
    for startInd in range(len(maxInds)):
        probMatrix = np.zeros((maxN**2,maxN**2),dtype='float32')
        probMatrix_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, probMatrix.nbytes)
        for n_A in range(maxN):
            for n_B in range(maxN):
                if [n_A,n_B] in maxInds and [n_A,n_B] != maxInds[startInd]:
                    continue
                probs_GPU = pyopencl.array.empty(queue,(n_A + 1,maxl_a + 1,n_B + 1,maxl_a + 1),dtype='float32')
                prob_init(probs_GPU,factMat_GPU,conditions['h_a'],conditions['h_A'],\
                conditions['k_Aa'],conditions['k_Ab'],conditions['maxl_a'],n_A,n_B,maxN)
                probs_GPU = pyopencl.clmath.exp(probs_GPU - pyopencl.array.max(probs_GPU).get())
                probs_GPU /= pyopencl.array.sum(probs_GPU).get()
                prg.fsp_add(queue,(probMatrix.shape[0],1),None,probMatrix_buf,probs_GPU.data,\
                np.uint16(n_A),np.uint16(n_B),np.uint16(maxl_a),np.uint16(maxN))
        probMatrix = np.empty_like(probMatrix)
        pyopencl.enqueue_copy(queue,probMatrix,probMatrix_buf)
        probMatrix = np.hstack((probMatrix,np.zeros((maxN**2,1))))
        probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
        probMatrix = np.linalg.matrix_power(probMatrix,inc)
        probInit = np.zeros(len(probMatrix))
        probInit[maxN*maxInds[startInd][1] + maxInds[startInd][0]] = 1.0
        dwellCumeProb = [0.0]
        while 1 - dwellCumeProb[-1] > maxTol:
            print(dwellCumeProb[-1])
            probInit = np.dot(probMatrix,probInit)
            dwellCumeProb.append(probInit[-1] - probInit[maxN*maxInds[startInd][1] + maxInds[startInd][0]])
            for ind in range(len(maxInds)):
                dwellCumeProb[-1] += probInit[maxN*maxInds[ind][1] + maxInds[ind][0]]
        dwellProbs.append(np.array(dwellCumeProb[1:]) - np.array(dwellCumeProb[:-1]))
    return dwellProbs

def rateCalc(lagrangeVals,maxl_a,n_A,n_B):
    """ Calculates effective production and degradation rates for a given set of MaxCal parameters """
    global maxN
    global factMat_GPU
    l_A = np.array([[[indA*np.ones(maxl_a + 1) \
    for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
    for indA in range(n_A + 1)])
    l_a = np.array([[[inda*np.ones(maxl_a + 1) \
    for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
    for indA in range(n_A + 1)])
    l_B = np.array([[[indB*np.ones(maxl_a + 1) \
    for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
    for indA in range(n_A + 1)])
    l_b = np.array([[[np.arange(maxl_a + 1) \
    for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
    for indA in range(n_A + 1)])
    probVals = pyopencl.array.empty(queue,(n_A + 1,maxl_a + 1,n_B + 1,maxl_a + 1),dtype='float32')
    prob_init(probVals,factMat_GPU,lagrangeVals[0],lagrangeVals[1],lagrangeVals[2],lagrangeVals[3],maxl_a,n_A,n_B,maxN)
    probVals = pyopencl.clmath.exp(probVals - pyopencl.array.max(probVals).get())
    probVals /= pyopencl.array.sum(probVals).get()
    prodRateA = np.sum(l_a*probVals.get())
    if n_A > 0:
        degRateA = np.sum((n_A - l_A)*probVals.get())
        degRateA /= n_A
    else:
        degRateA = float('NaN')
    prodRateB = np.sum(l_b*probVals.get())
    if n_B > 0:
        degRateB = np.sum((n_B - l_B)*probVals.get())
        degRateB /= n_B
    else:
        degRateB = float('NaN')
    return prodRateA, degRateA, prodRateB, degRateB

def feedbackCalc(lagrangeVals,maxl_a,equilProb):
    """ Calculates the effective feedback metric for a given of MaxCal parameters """
    global maxN
    global factMat_GPU
    if len(equilProb) == 0:
        equilProb = maxCalEquil(lagrangeVals,maxl_a)
    avgl_a,avgl_A,avgl_b,avgl_B,avgl_a2,avgl_A2,avgl_b2,avgl_B2,avgl_al_A,\
    avgl_bl_A,avgl_al_B,avgl_bl_B = tuple([0.0 for ind in range(12)])
    for n_A in range(len(equilProb)):
        for n_B in range(len(equilProb[n_A])):
            probsMC = pyopencl.array.empty(queue,(n_A + 1,maxl_a + 1,n_B + 1,maxl_a + 1),dtype='float32')
            prob_init(probsMC,factMat_GPU,lagrangeVals[0],lagrangeVals[1],lagrangeVals[2],lagrangeVals[3],maxl_a,n_A,n_B,maxN)
            probsMC = pyopencl.clmath.exp(probsMC - pyopencl.array.max(probsMC).get())
            probsMC /= pyopencl.array.sum(probsMC).get()
            probsMC = probsMC.get()
            l_AVals = np.array([[[indA*np.ones(maxl_a + 1) \
            for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
            for indA in range(n_A + 1)])
            l_aVals = np.array([[[inda*np.ones(maxl_a + 1) \
            for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
            for indA in range(n_A + 1)])
            l_BVals = np.array([[[indB*np.ones(maxl_a + 1) \
            for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
            for indA in range(n_A + 1)])
            l_bVals = np.array([[[np.arange(maxl_a + 1) \
            for indB in range(n_B + 1)] for inda in range(maxl_a + 1)] \
            for indA in range(n_A + 1)])
            avgl_a += equilProb[n_A,n_B]*np.sum(probsMC*l_aVals)
            avgl_A += equilProb[n_A,n_B]*np.sum(probsMC*l_AVals)
            avgl_b += equilProb[n_A,n_B]*np.sum(probsMC*l_bVals)
            avgl_B += equilProb[n_A,n_B]*np.sum(probsMC*l_BVals)
            avgl_a2 += equilProb[n_A,n_B]*np.sum(probsMC*l_aVals*l_aVals)
            avgl_A2 += equilProb[n_A,n_B]*np.sum(probsMC*l_AVals*l_AVals)
            avgl_b2 += equilProb[n_A,n_B]*np.sum(probsMC*l_bVals*l_bVals)
            avgl_B2 += equilProb[n_A,n_B]*np.sum(probsMC*l_BVals*l_BVals)
            avgl_al_A += equilProb[n_A,n_B]*np.sum(probsMC*l_aVals*l_AVals)
            avgl_bl_A += equilProb[n_A,n_B]*np.sum(probsMC*l_bVals*l_AVals)
            avgl_al_B += equilProb[n_A,n_B]*np.sum(probsMC*l_aVals*l_BVals)
            avgl_bl_B += equilProb[n_A,n_B]*np.sum(probsMC*l_bVals*l_BVals)
    stDevl_a = (avgl_a2 - avgl_a**2.0)**0.5
    stDevl_A = (avgl_A2 - avgl_A**2.0)**0.5
    stDevl_b = (avgl_b2 - avgl_b**2.0)**0.5
    stDevl_B = (avgl_B2 - avgl_B**2.0)**0.5
    feedbackAa = (avgl_al_A - avgl_a*avgl_A)/(stDevl_a*stDevl_A)
    feedbackAb = (avgl_bl_A - avgl_b*avgl_A)/(stDevl_b*stDevl_A)
    feedbackBa = (avgl_al_B - avgl_a*avgl_B)/(stDevl_a*stDevl_B)
    feedbackBb = (avgl_bl_B - avgl_b*avgl_B)/(stDevl_b*stDevl_B)
    return feedbackAa, feedbackAb, feedbackBa, feedbackBb



def cudaMatPower(matrix, power):
    matpower = cp.array(np.copy(matrix))
    holder = cp.array(np.copy(matrix))
    for i in range(power-1):
        matpower = cp.dot(matpower, holder) 
    matpower = cp.asnumpy(matpower)
    return matpower

def maxCal_mle(lagrangeVals):
    """ Uses Finite State Projection to calculate the likelihood of """
    """ an input trajectory occurring in Maximum Caliber. """
    """ lagrangeVals = ['h_a','h_A','k_Aa','k_Ab'] """
    global probs
    global maxl_a
    global maxN
    global numIterations
    global factMat_GPU
    global ctx
    global mf
    global timeVals
    
    probMatrix = np.zeros((maxN**2,maxN**2),dtype='float32')
    probMatrix_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, probMatrix.nbytes)
    for n_A in range(maxN):
        for n_B in range(maxN):
            probs_GPU = pyopencl.array.empty(queue,(n_A + 1,maxl_a + 1,n_B + 1,maxl_a + 1),dtype='float32')
            prob_init(probs_GPU,factMat_GPU,lagrangeVals[0],lagrangeVals[1],lagrangeVals[2],lagrangeVals[3],maxl_a,n_A,n_B,maxN)
            probs_GPU = pyopencl.clmath.exp(probs_GPU - pyopencl.array.max(probs_GPU).get())
            probs_GPU /= pyopencl.array.sum(probs_GPU).get()
            prg.fsp_add(queue,(probMatrix.shape[0],1),None,probMatrix_buf,probs_GPU.data,\
            np.uint16(n_A),np.uint16(n_B),np.uint16(maxl_a),np.uint16(maxN))
    probMatrix = np.empty_like(probMatrix)
    pyopencl.enqueue_copy(queue,probMatrix,probMatrix_buf)
    probMatrix = np.hstack((probMatrix,np.zeros((maxN**2,1))))
    probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
    startTime = datetime.datetime.now()
    probMatrixCP = cp.array(probMatrix, cp.float32)
    """ GPU Matrix Power Function  """
    tempMatrix = cp.linalg.matrix_power(probMatrixCP, numIterations)
    tempMatrix = cp.asnumpy(tempMatrix)
    endTime = datetime.datetime.now()
    timeVals.append((endTime - startTime).total_seconds())
    loglike = -1*np.nansum(np.nansum(np.log(tempMatrix)*probs.toarray()))
    tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
    tempData.write('h_a = ' + str(round(lagrangeVals[0],3)) + ', h_A = ' + str(round(lagrangeVals[1],3)) + \
    ', K_Aa = ' + str(round(lagrangeVals[2],4)) + ', K_Ab = ' + str(round(lagrangeVals[3],4)) + \
    ', M = ' + str(maxl_a) + ', loglike = ' + str(round(loglike,1)) + '\n')
    tempData.close()
    return loglike

""" Defining Reaction Rates """

#conditions_Gill = conditionsInitGill(g,g_pro,g_rep,g_prorep,d,p,r,k_f,k_b,exclusive,\
#n_A_init,n_a_init,n_alpha_init,n_B_init,n_b_init,n_beta_init,inc,numSteps,numTrials)
conditions_Gill = conditionsInitGill(0.5,0.5,0.0025,0.0025,0.5,0.02,0.001,\
3.5e-06,2.0e-05,True,5,5,0,5,5,0,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" If simulations are already present, load them """
""" If not, create them using the Gillespie function above """

tempVars = np.load('Simulations/' + fname)
n_A_Gill = tempVars['n_A_Gill']
n_B_Gill = tempVars['n_B_Gill']
n_A_Gill = n_A_Gill[:,range(0,24*3600*numDays,timeInc)]
n_B_Gill = n_B_Gill[:,range(0,24*3600*numDays,timeInc)]
del tempVars

""" Running Metrics on Gillespie Input Trajectories """

simHist_Gill = np.zeros((maxN,maxN))
for numTrial in range(len(n_A_Gill)):
    simHist_Gill += np.histogram2d(n_A_Gill[numTrial],n_B_Gill[numTrial],bins=np.arange(-0.5,maxN))[0]
del numTrial
simHist_Gill = simHist_Gill/np.sum(simHist_Gill)
maxInds_Gill = peakVals(simHist_Gill[:75,:75],5,0.0001)
if maxInds_Gill[1][1] != 0:
    maxInds_Gill.append(maxInds_Gill.pop(1))
totEntropy_Gill,stateEntropies_Gill,macroEntropy_Gill,dwellVals_Gill = entropyStats(n_A_Gill,n_B_Gill,maxInds_Gill)
avgDwells_Gill = []
avgTotDwell_Gill = []
for ind in range(len(dwellVals_Gill)):
    avgDwells_Gill.append(np.average(dwellVals_Gill[ind])*conditions_Gill['inc'])
    avgTotDwell_Gill.extend(dwellVals_Gill[ind])
del ind
avgTotDwell_Gill = np.average(avgTotDwell_Gill)*conditions_Gill['inc']
if type(numIterations) == str:
    numIterations = int(round(avgTotDwell_Gill/conditions_Gill['inc']))

""" Calculating the Transition Frequencies for Maximum Likelihood Calculations """

probs = scipy.sparse.csc_matrix((np.ones(np.size(n_A_Gill[:,numIterations::numIterations])),\
(maxN*n_B_Gill[:,numIterations::numIterations].reshape(np.size(n_B_Gill[:,numIterations::numIterations])) + \
n_A_Gill[:,numIterations::numIterations].reshape(np.size(n_A_Gill[:,numIterations::numIterations])),\
maxN*n_B_Gill[:,:-numIterations:numIterations].reshape(np.size(n_B_Gill[:,:-numIterations:numIterations])) + \
n_A_Gill[:,:-numIterations:numIterations].reshape(np.size(n_A_Gill[:,:-numIterations:numIterations])))),shape=(maxN**2 + 1,maxN**2 + 1))
del n_A_Gill, n_B_Gill

""" Fitting Optimal Parameters Using Maximum Likelihood Procedure """

loglike = float('NaN')*np.ones(maxMaxl_a)

""" DOUBLE CHECK THAT THIS EXISTS!!! """
tempVars = np.load('SampleParameters_ToggleMaxCal.npz')
""" DOUBLE CHECK THAT THIS EXISTS!!! """

finalGuess = tempVars['finalGuess'][:maxMaxl_a,:]
del tempVars

startTime_Tot = datetime.datetime.now()
timeVals = []

for maxl_a in range(minMaxl_a,maxMaxl_a + 1):
    tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
    tempData.write('Max l_a = ' + str(maxl_a) + '\n')
    tempData.close()
    """ Using a previous fit as a starting point for the minimization algorithm """
    """ Not 100% objective, but will make the fit go much faster... """
    bestGuess = finalGuess[maxl_a - 1]
    res = minimize(maxCal_mle,bestGuess,method='nelder-mead',tol=0.1,options={'disp':True,'maxiter':500})
    loglike[maxl_a - 1] = res['fun']
    finalGuess[maxl_a - 1] = res['x']
    """ Saving a checkpoint as we go... """
    np.savez_compressed(fout, \
    conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess, timeVals=timeVals)
for numTry in range(2):
    for maxl_a in range(maxMaxl_a - 1,minMaxl_a - 1,-1):
        tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
        tempData.write('Max l_a = ' + str(maxl_a) + '\n')
        tempData.close()
        """ Using different starting seeds to find the truly optimized solution... """
        bestGuess = finalGuess[maxl_a] + np.array([0.2*np.random.rand(1)[0] - 0.1,\
        0.2*np.random.rand(1)[0] - 0.1,0.02*np.random.rand(1)[0] - 0.01,0.02*np.random.rand(1)[0] - 0.01])
        res = minimize(maxCal_mle,bestGuess,method='nelder-mead',tol=0.1,options={'disp':True,'maxiter':500})
        """ Only store the result if it beats the previous results... """
        if res['fun'] < loglike[maxl_a - 1] or np.isnan(loglike[maxl_a - 1]):
            loglike[maxl_a - 1] = res['fun']
            finalGuess[maxl_a - 1] = res['x']
        """ Saving a checkpoint as we go... """
        np.savez_compressed(fout, \
        conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess, timeVals=timeVals)
    for maxl_a in range(minMaxl_a + 1,maxMaxl_a + 1):
        tempData = open('ProgressReport_ToggleMaxCal_' + str(simNum) + '.txt','a')
        tempData.write('Max l_a = ' + str(maxl_a) + '\n')
        tempData.close()
        """ Using different starting seeds to find the truly optimized solution... """
        bestGuess = finalGuess[maxl_a - 2] + np.array([0.2*np.random.rand(1)[0] - 0.1,\
        0.2*np.random.rand(1)[0] - 0.1,0.02*np.random.rand(1)[0] - 0.01,0.02*np.random.rand(1)[0] - 0.01])
        res = minimize(maxCal_mle,bestGuess,method='nelder-mead',tol=0.1,options={'disp':True,'maxiter':500})
        """ Only store the result if it beats the previous results... """
        if res['fun'] < loglike[maxl_a - 1] or np.isnan(loglike[maxl_a - 1]):
            loglike[maxl_a - 1] = res['fun']
            finalGuess[maxl_a - 1] = res['x']
        """ Saving a checkpoint as we go... """
        np.savez_compressed(fout, \
        conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess, timeVals=timeVals)
del numTry

endTime_Tot = datetime.datetime.now()
fitTime = endTime_Tot - startTime_Tot
del startTime_Tot, endTime_Tot

""" Running Metrics on Inferred MaxCal Parameters """

maxl_a = int(np.where(loglike == np.nanmin(loglike))[0][0]) + 1
bestGuess = finalGuess[maxl_a - 1]
equilProb_MaxCal = maxCalEquil(bestGuess,maxl_a)
maxInds_MaxCal = peakVals(equilProb_MaxCal[:75,:75],5,0.001)
if maxInds_MaxCal[1][1] != 0:
    maxInds_MaxCal.append(maxInds_MaxCal.pop(1))
prodRates = float('NaN')*np.ones((len(maxInds_MaxCal),2))
degRates = float('NaN')*np.ones((len(maxInds_MaxCal),2))
for ind in range(len(maxInds_MaxCal)):
    prodRates[ind,0],degRates[ind,0],prodRates[ind,1],degRates[ind,1] = \
    rateCalc(bestGuess,maxl_a,maxInds_MaxCal[ind][0],maxInds_MaxCal[ind][1])
del ind
prodRates /= timeInc
degRates /= timeInc
feedbackAa,feedbackAb,feedbackBa,feedbackBb = feedbackCalc(bestGuess,maxl_a,equilProb_MaxCal)

np.savez_compressed(fout, \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess, \
maxl_a=maxl_a, bestGuess=bestGuess, equilProb_MaxCal=equilProb_MaxCal, \
maxInds_MaxCal=maxInds_MaxCal, prodRates=prodRates, degRates=degRates, \
feedbackAa=feedbackAa, feedbackAb=feedbackAb, feedbackBa=feedbackBa, \
feedbackBb=feedbackBb, timeVals=timeVals, fitTime=fitTime)

#conditions_MaxCal = conditionsInitMaxCal(h_a,h_A,k_Aa,k_Ab,maxl_a,\
#n_A_init,l_a_init,l_iA_init,n_B_init,l_b_init,l_iB_init,inc,numSteps,numTrials)
conditions_MaxCal = conditionsInitMaxCal(bestGuess[0],bestGuess[1],bestGuess[2],bestGuess[3],maxl_a,\
5,0,5,5,0,5,timeInc,int((24*3600*numDays)/timeInc),numTrials)

n_A_MaxCal = []
n_B_MaxCal = []
for numTrial in range(numTrials):
    randNum = np.random.uniform(low=0,high=1)
    totProb = 0.0
    for n_A_init in range(maxN):
        for n_B_init in range(maxN):
            totProb += equilProb_MaxCal[n_A_init][n_B_init]
            if totProb > randNum:
                break
        if totProb > randNum:
            break
    n_A_MaxCal.append([n_A_init])
    n_B_MaxCal.append([n_B_init])
    del n_A_init, n_B_init, totProb, randNum
del numTrial
l_a_MaxCal = np.zeros((conditions_MaxCal['numTrials'],1)).tolist()
l_iA_MaxCal = np.copy(n_A_MaxCal).tolist()
l_b_MaxCal = np.zeros((conditions_MaxCal['numTrials'],1)).tolist()
l_iB_MaxCal = np.copy(n_B_MaxCal).tolist()
n_A_MaxCal,l_a_MaxCal,l_iA_MaxCal,n_B_MaxCal,l_b_MaxCal,l_iB_MaxCal = \
maxCalSim(conditions_MaxCal,n_A_MaxCal,l_a_MaxCal,l_iA_MaxCal,n_B_MaxCal,l_b_MaxCal,l_iB_MaxCal)
n_A_MaxCal = np.array(n_A_MaxCal)
n_B_MaxCal = np.array(n_B_MaxCal)
totEntropy_MaxCal,stateEntropies_MaxCal,macroEntropy_MaxCal,dwellVals_MaxCal = entropyStats(n_A_MaxCal,n_B_MaxCal,maxInds_MaxCal)
avgDwells_MaxCal = []
avgTotDwell_MaxCal = []
for ind in range(len(dwellVals_MaxCal)):
    avgDwells_MaxCal.append(np.average(dwellVals_MaxCal[ind])*conditions_MaxCal['inc'])
    avgTotDwell_MaxCal.extend(dwellVals_MaxCal[ind])
del ind
avgTotDwell_MaxCal = np.average(avgTotDwell_MaxCal)*conditions_MaxCal['inc']

dwellProbs_MaxCal = dwellHistMaxCal(conditions_MaxCal,maxInds_MaxCal,0.001,30)
dwellInds_MaxCal = []
avgDwells_MaxCalFSP = []
for ind in range(len(dwellProbs_MaxCal)):
    dwellInds_MaxCal.append(conditions_MaxCal['inc']*np.arange(0,30*len(dwellProbs_MaxCal[ind]),30))
    avgDwells_MaxCalFSP.append(sum(dwellInds_MaxCal[ind]*dwellProbs_MaxCal[ind]))
del ind

np.savez_compressed(fout, \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess, \
maxl_a=maxl_a, bestGuess=bestGuess, equilProb_MaxCal=equilProb_MaxCal, \
maxInds_MaxCal=maxInds_MaxCal, prodRates=prodRates, degRates=degRates, \
feedbackAa=feedbackAa, feedbackAb=feedbackAb, feedbackBa=feedbackBa, \
feedbackBb=feedbackBb, conditions_MaxCal=conditions_MaxCal, n_A_MaxCal=n_A_MaxCal, \
l_a_MaxCal=l_a_MaxCal, l_iA_MaxCal=l_iA_MaxCal, n_B_MaxCal=n_B_MaxCal, \
l_b_MaxCal=l_b_MaxCal, l_iB_MaxCal=l_iB_MaxCal, \
stateEntropies_MaxCal=stateEntropies_MaxCal, macroEntropy_MaxCal=macroEntropy_MaxCal, \
totEntropy_MaxCal=totEntropy_MaxCal, dwellVals_MaxCal=dwellVals_MaxCal, \
avgDwells_MaxCal=avgDwells_MaxCal, avgTotDwell_MaxCal=avgTotDwell_MaxCal, \
dwellProbs_MaxCal=dwellProbs_MaxCal, dwellInds_MaxCal=dwellInds_MaxCal, \
avgDwells_MaxCalFSP=avgDwells_MaxCalFSP, timeVals=timeVals, fitTime=fitTime)


