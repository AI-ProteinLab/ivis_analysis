import mdtraj as md
import numpy as np
from sklearn.cluster import KMeans
import math
from itertools import combinations 

def cluster_rmsd(gap, reduced_data):
    '''
    do kmneans clustering and do calculation of rmsd for each cluster
    '''

    def get_traj(gap):
        '''
        get the md trajectory
        '''
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
        combined = [dark_1[::gap], dark_2[::gap], dark_3[::gap], dark_fmc_1[::gap], dark_fmc_2[::gap], dark_fmc_3[::gap], light_fmn_1[::gap], light_fmn_2[::gap], light_fmn_3[::gap], light_1[::gap], light_2[::gap], light_3[::gap]]
        return combined


    def get_index(num):
        '''
        convert the index in cluster labels into index that match the trajectory
        '''
        loc_traj_0 = num[0] // math.ceil(9990 / gap)
        loc_frame_0 = num[0] - math.ceil(9990 / gap) * loc_traj_0
        loc_traj_1 = num[1] // math.ceil(9990 / gap)
        loc_frame_1 = num[1] - math.ceil(9990 / gap) * loc_traj_1
        return loc_traj_0, loc_frame_0, loc_traj_1, loc_frame_1


    def _rmsd(index):
        '''
        calculate the rmsd of two frames given index
        '''
        index_11, index_12, index_21, index_22 = index
        return md.rmsd(combined[index_11][index_12], combined[index_21][index_22])[0]


    def calculate_rmsd(index, gap):
        '''
        calculate the rmsd by convert the cluster labels into pairwise labels and
        do calculation
        '''
        result = []
        for cluster in index:
            com = list(combinations(cluster, r = 2))
            pair = list(map(get_index, com))
            accumulated = list(map(_rmsd, pair))
            if len(accumulated) > 0:
                result.append(sum(accumulated) / len(accumulated))
            #print(result)
        return result


    def get_cluster(data, num_of_cluster=1000):
        '''
        get cluster labels given embedding data
        '''
        kmeans = KMeans(n_clusters = num_of_cluster, random_state=0).fit(data)
        index = [[] for i in range(num_of_cluster)]
        for i in range(len(kmeans.labels_)):
            index[kmeans.labels_[i]].append(i)
        return index


    global combined
    combined = get_traj(gap)
    index = get_cluster(reduced_data)
    result = calculate_rmsd(index, gap)
    return result


