import mdtraj as md
import numpy as np
import math

def get_traj_data(gap, cutoff, start = 0):

    ######## load data #########
    dark_1 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkk/7-nvt_production/01/dimer_5dkk_01_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkk/7-nvt_production/01/ca.psf")
    dark_2 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkk/7-nvt_production/02/dimer_5dkk_02_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkk/7-nvt_production/01/ca.psf")
    dark_3 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkk/7-nvt_production/03/dimer_5dkk_03_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkk/7-nvt_production/01/ca.psf")
    light_1 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkl/7-nvt_production/01/dimer_5dkl_01_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkl/7-nvt_production/01/ca.psf")
    light_2 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkl/7-nvt_production/02/dimer_5dkl_02_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkl/7-nvt_production/01/ca.psf")
    light_3 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkl/7-nvt_production/03/dimer_5dkl_03_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_5dkl/7-nvt_production/01/ca.psf")
    dark_fmc_1 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_dark_fmc/7-nvt_production/01/dimer_dark_fmc_01_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_dark_fmc/7-nvt_production/01/ca.psf")
    dark_fmc_2 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_dark_fmc/7-nvt_production/02/dimer_dark_fmc_02_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_dark_fmc/7-nvt_production/01/ca.psf")
    dark_fmc_3 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_dark_fmc/7-nvt_production/03/dimer_dark_fmc_03_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_dark_fmc/7-nvt_production/01/ca.psf")
    light_fmn_1 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_light_fmn/7-nvt_production/01/dimer_light_fmn_01_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_light_fmn/7-nvt_production/01/ca.psf")
    light_fmn_2 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_light_fmn/7-nvt_production/02/dimer_light_fmn_02_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_light_fmn/7-nvt_production/01/ca.psf")
    light_fmn_3 = md.load("/users/haot/scratch/research/PtAu1a/dimer/dimer_light_fmn/7-nvt_production/03/dimer_light_fmn_03_50000_tot_ca.dcd", top="/users/haot/scratch/research/PtAu1a/dimer/dimer_light_fmn/7-nvt_production/01/ca.psf")
    
    feature = 254
    index = [[i, j] for i in range(0, feature) for j in range(i+1, feature)]

    ###### compute distance ########
    distance_dark_1 = md.compute_distances(dark_1, index)
    distance_dark_2 = md.compute_distances(dark_2, index)
    distance_dark_3 = md.compute_distances(dark_3, index)
    distance_light_1 = md.compute_distances(light_1, index)
    distance_light_2 = md.compute_distances(light_2, index)
    distance_light_3 = md.compute_distances(light_3, index)
    distance_dark_fmc_1 = md.compute_distances(dark_fmc_1, index)
    distance_dark_fmc_2 = md.compute_distances(dark_fmc_2, index)
    distance_dark_fmc_3 = md.compute_distances(dark_fmc_3, index)
    distance_light_fmn_1 = md.compute_distances(light_fmn_1, index)
    distance_light_fmn_2 = md.compute_distances(light_fmn_2, index)
    distance_light_fmn_3 = md.compute_distances(light_fmn_3, index)

    ###### combine data ########
    num_of_sample = math.ceil(9990 / gap)
    X = np.concatenate((distance_dark_1[start::gap], distance_dark_2[start::gap], distance_dark_3[start::gap], distance_dark_fmc_1[start::gap], distance_dark_fmc_2[start::gap], distance_dark_fmc_3[start::gap], distance_light_fmn_1[start::gap], distance_light_fmn_2[start::gap], distance_light_fmn_3[start::gap], distance_light_1[start::gap], distance_light_2[start::gap], distance_light_3[start::gap]), axis = 0)
    X = np.where(X <= cutoff, X, 0)
    y = np.array([[0] * num_of_sample * 3 + [1] * num_of_sample * 3+ [2] * num_of_sample * 3 + [3] * num_of_sample * 3])
    y = y.reshape(-1, 1)
    return X, y, gap, cutoff, num_of_sample

