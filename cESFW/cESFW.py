
''' Continuous Entropy Sort Feature Weighting (cESFW) '''

''' Here, Radley et al. provide the code for their cESFW software.
cESFW is a feature weighting algorithm designed for distinguishing between informative and uninformative features
in high dimensional data. cESFW is founded on the principles of Entropy Sorting, which Radley et al. have previously
described (1). 

cESFW was primarily designed with single cell RNA sequencing data in mind, but should be applicable to many different types of
high dimensional data, where the user would like to remove uninformative features. By focussing on significantly
correlating features in a dataset, it is hoped that downstream analysis of complex data should be improved.

(1) Entropy sorting of single-cell RNA sequencing (scRNA-seq) data reveals the inner cell mass in the human pre-implantation embryo. Arthur Radley, 
Elena Corujo-Simon, Jennifer Nichols, Austin Smith, Sara-Jane Dunn. Stem Cell Reports. 2023.'''

##### Dependencies #####

import numpy as np
import pandas as pd
from functools import partial 
import multiprocess
import scipy.sparse
from p_tqdm import p_map

##### Dependencies #####

##### Converting a scaled matrix into a memory and compute power efficient format. #####

# The primary input for cESFW is a 2D matrix where each row is a sample and each columns is a feature. For scRNA-seq data
# the cells would be the samples and the genes would be the features. Each feature must be normalised/scaled such that all the values
# are between 0 and 1. How each feature is scaled such that is values are between 0 and 1 is determined by the user, since this is dataset
# dependent, but we do provide examples in our workflows for applying cESFW to scRNA-seq data. 
# Given the scaled matrix, cESFW will use the following function to extract all the non-zero values into a single vector. We can do this
# because ES calculations can easily ignore 0 values in the data. For sparse data like scRNA-seq data, this dramatically reduces the memory
# required, and the number of calculations that need to be carried out. For relitively dense data, this step will still need to be carried
# out to use cESFW, but will provide little benifit computationally.

### This function converts a scaled matrix into a computationally efficient object.
## path: A string path pre-designated folder to deposit the computationally efficient object. E.g. "/mnt/c/Users/arthu/Test_Folder/"
## Scaled_Matrix: The high dimensional DataFrame whose features have been scaled to values between 0 and 1. Format must be a Pandas DataFrame.
## Min_Minority_State_Cardinality: The minimum value of the total minority state mass that a feature contains before it will be automatically
# removed from the data, and hence analysis.

def Create_ESFW_Objects(path, Scaled_Matrix, Min_Minority_State_Cardinality = 10):
    # Extract the feature names/IDs.
    All_Feature_IDs = Scaled_Matrix.columns
    # Convert data to a sparse matrix for manipulation.
    Scaled_Matrix = scipy.sparse.csr_matrix(Scaled_Matrix.values)
    # Calculate feature minority state masses.
    Feature_Sums = np.asarray(Scaled_Matrix.sum(axis=0))[0]
    Feature_Sums[np.isnan(Feature_Sums)] = 0
    # Indeitfy features that pass the Min_Minority_State_Cardinality threshold.
    Keep_Features = np.where(Feature_Sums >= Min_Minority_State_Cardinality)[0]
    Ignore = np.where(Feature_Sums >= (Scaled_Matrix.shape[0]-Min_Minority_State_Cardinality))[0]
    Keep_Features = np.delete(Keep_Features,np.where(np.isin(Keep_Features,Ignore))[0])
    # Subset down to features that pass the Min_Minority_State_Cardinality threshold. 
    if Keep_Features.shape[0] < Scaled_Matrix.shape[1]:
        print("Ignoring " + str(Scaled_Matrix.shape[1]-Keep_Features.shape[0]) + " features which are below the Min_Minority_State_Cardinality")
        Scaled_Matrix = Scaled_Matrix[:,Keep_Features]
    # Track the number of samples and features in the subsetted data.
    Sample_Cardinality = Scaled_Matrix.shape[0]
    Feature_Cardinality = Scaled_Matrix.shape[1]
    # Convert Scaled_Matrix into Minority_Group_Matrix. I.e. ensure that features don't have a mass greater than half the sample cardinality.
    Minority_State_Masses = np.asarray(Scaled_Matrix.sum(axis=0).astype("f"))[0]
    Switch_State_Inidicies = np.where(Minority_State_Masses >= (Sample_Cardinality/2))[0]
    Minority_State_Masses[Switch_State_Inidicies] = Sample_Cardinality - Minority_State_Masses[Switch_State_Inidicies]  
    # Switch out minority/maority states in the scaled matrix where necessary.
    if Switch_State_Inidicies.shape[0] > 0:
        Replace = Scaled_Matrix[:,Switch_State_Inidicies].todense()
        Replace = 1 - Replace
        Scaled_Matrix[:,Switch_State_Inidicies] = scipy.sparse.csr_matrix(Replace)
    # Print the number of samples and features in the processed data
    print("Number of samples: " + str(Sample_Cardinality))
    print("Number of features: " + str(Feature_Cardinality))
    Used_Features = All_Feature_IDs[Keep_Features]
    # Extract the non-zero inds and values of the 2D matrix into a single vector.
    Scaled_Matrix_Non_Zero_Inds = Scaled_Matrix.nonzero()
    Scaled_Matrix_Non_Zero_Values = np.asarray(Scaled_Matrix[Scaled_Matrix_Non_Zero_Inds])[0]
    # Save memory efficient information and objects
    np.save(path + "Minority_State_Masses.npy",Minority_State_Masses)
    np.save(path + "Used_Features.npy",Used_Features)
    np.save(path + "Sample_Cardinality.npy",Sample_Cardinality)
    np.save(path + "Feature_Cardinality.npy",Feature_Cardinality)
    np.save(path + "Scaled_Matrix_Non_Zero_Inds.npy",Scaled_Matrix_Non_Zero_Inds)
    np.save(path + "Scaled_Matrix_Non_Zero_Values.npy",Scaled_Matrix_Non_Zero_Values)


##### Now that we have our compute efficient object, we can calculate the Entropy Sort Score (ESS) an Error Potential (EP) values,
# pairwise for each feature. #####

### Parallel_Calculate_ESS_EPs is the main wrapper function which will output the feature pairwise ESS and EP matrix for a given
# scaled dataframe.
## path: A string path pre-designated folder to deposit the computationally efficient object. E.g. "/mnt/c/Users/arthu/Test_Folder/"
# This must be the sampe path that was used to deposit/save our compute efficient objects with the Create_ESFW_Objects function.
## Use_Cores: The number of cores to use. A default setting on -1 means the software will use the number of cores it detects on
# the machine, minus 1. Users can manually set any value to please.
## EP_Masked_ESSs: This paramter deterimes whether the function will output the pairwise ESS and EP matricies seperately, or whether
# it will automatically mask the ESS matrix by setting all indicies that are less than or equal to 0 in the EP matrix, to 0 in the ESS matrix.
# Default is True, and leads to a single matrix being output by the function. If not True, a seperate ESS and EP matrix are output
# from the function.

def Parallel_Calculate_ESS_EPs(path,Use_Cores=-1,EP_Masked_ESSs=True):
    ## Load compute efficient cESFW objects.
    Minority_State_Masses = np.load(path + "Minority_State_Masses.npy")
    Sample_Cardinality = np.load(path + "Sample_Cardinality.npy")
    Feature_Cardinality = np.load(path + "Feature_Cardinality.npy")
    global Scaled_Matrix_Non_Zero_Inds
    Scaled_Matrix_Non_Zero_Inds = np.load(path + "Scaled_Matrix_Non_Zero_Inds.npy")
    global Scaled_Matrix_Non_Zero_Values
    Scaled_Matrix_Non_Zero_Values = np.load(path + "Scaled_Matrix_Non_Zero_Values.npy")
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(Feature_Cardinality)
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    if __name__ == '__main__':
        with np.errstate(divide='ignore',invalid='ignore'):
            Results = p_map(partial(Calculate_ESS_EPs,Sample_Cardinality=Sample_Cardinality,Feature_Cardinality=Feature_Cardinality,Minority_State_Masses=Minority_State_Masses), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results
    Results = np.asarray(Results)
    ESSs = Results[:,0]
    EPs = Results[:,1]
    ## Use the symmetic properties of Entropy Sorting and the maximum entropy principle to create the final matricies.
    # In very rare scenarios where the feature cardinalities are identical but not perfectly overlapping, the features will both
    # see each other as the larger feature and hence the ESSs and EPs will be double calculated. Hence, we must account for these double
    # counts.
    ## Symmetric ESSs
    Double_Counts = (ESSs != 0).astype("i") + (ESSs.T != 0).astype("i")
    Double_Counts = np.where(Double_Counts == 2)
    ESSs = ESSs+ESSs.T
    ESSs[Double_Counts] = ESSs[Double_Counts] / 2
    ## Symmetric EPs
    Double_Counts = (EPs != 0).astype("i") + (EPs.T != 0).astype("i")
    Double_Counts = np.where(Double_Counts == 2)
    EPs = EPs+EPs.T
    EPs[Double_Counts] = EPs[Double_Counts] / 2
    # Return results
    if EP_Masked_ESSs != True:
        return ESSs, EPs
    if EP_Masked_ESSs == True:
        EPs[EPs < 0] = 0
        return ESSs * EPs


### Calculate ESSs and EPs
# This is the main function for generating the parwise ESS and EP matricies. Given a fixed feature, defined by the feature in of the parallel
# function, Calculate_ESS_EPs will identify all the features where the fixed feature is the reference feature (RF), as defined by it's minority state
# mass being less than that of another feature. Calculate_ESS_EPs then looks at each error scenario where the fixed feature can be the RF
# and calculates the relevent ESS and EP values.
# Importantly, because we are calculating the entire pariwise ESS and EP matrix, every pairwise RF/QF arrangment according to the maximum
# entropy principle will be calculated at least once. To get the full matrix, we then simply have to add the ESS and EP output matricies
# transpose of themselved. 
# The alternative would be to also calculate teh ESS and EPs for the fixed feature when it acts the te query feature
# QF. This would double the number of calculations, but remove the need to add the transpose of the outputs to themselves. Note that in
# the Calculate_Individual_ESS_EPs function in this code, this is exactly what we do, because we cannot exploit the symmetrical nature of
# the ESSs/EPs when we are only claculating them for a single feature. However, both apporaches give the same result, demonstrating they are
# equivalent.

## Feature_Ind: The current fixed feature for which we are calculating ESS and EP values against all other features. Provided by the 
# Parallel_Calculate_ESS_EPs function as it itterates through every feature.
# Sample_Cardinality: Number of samples in the data.
# Feature_Cardinality: Number of features in the data.
# Minority_State_Masses: The minority state masses of each feature in the data.

def Calculate_ESS_EPs(Feature_Ind,Sample_Cardinality,Feature_Cardinality,Minority_State_Masses):
    ## Initiate a deposit for the results.
    Results = []
    ## Initiate the ESS and EP storage vectors for each of the 3 error scenario that we will encounter.
    # EPs will be stored in an array called Saved_EPs to differentiate it from the EPs calculated in each error scenario.
    ESSs = np.zeros((3,Minority_State_Masses.shape[0]))
    Saved_EPs = np.zeros((3,Minority_State_Masses.shape[0]))
    ## We must re-create the fixed feature so that we can calculate its overlap with all otehr features in the data,
    # using out compute efficient methodology. If someone can work out how to calculate the overlaps in a way that doesn't require us to
    # re-expand the fixed feature indicies, it will mitigate the whole software compute time scaling linearly with the number of samples in
    # the data.
    Fixed_Feature = np.zeros(Sample_Cardinality)
    Fixed_Feature_Non_Zero_Inds = np.where(Scaled_Matrix_Non_Zero_Inds[1] == Feature_Ind)[0]  
    Fixed_Feature[Scaled_Matrix_Non_Zero_Inds[0][Fixed_Feature_Non_Zero_Inds]] = Scaled_Matrix_Non_Zero_Values[Fixed_Feature_Non_Zero_Inds]
    # Extract all the fixed future non-zero values that overlap with non-zero values of every other feature in the data
    Fixed_Feature_Non_Zero_Values = Fixed_Feature[Scaled_Matrix_Non_Zero_Inds[0]]
    # Find the minimum value of all the non-zero overlaps. This is where a lot of the speed up comes from, because the overlap with 0 values
    # will always be 0, so why bother calculating them?
    Overlaps = np.minimum(Fixed_Feature_Non_Zero_Values,Scaled_Matrix_Non_Zero_Values)
    Non_Zero_Overlaps = np.where(Overlaps != 0)[0]
    # Sum the overlaps of the fixed feature with every other feature.
    Overlaps = np.bincount(Scaled_Matrix_Non_Zero_Inds[1][Non_Zero_Overlaps], Overlaps[Non_Zero_Overlaps],minlength=Feature_Cardinality)
    # Adjust for float error between the earlier Minority_State_Masses caclulation and this calculation by substituting in the
    # Minority_State_Masses value. This really is a tiny float error and just neatens up the final results a little bit.
    Overlaps[Feature_Ind] = Minority_State_Masses[Feature_Ind]
    Fixed_Feature_Permutable_Cardinality = Minority_State_Masses[Feature_Ind]
    # Identify all the other features that have higher minority state masses than the current fixed feature.
    Higher_Minority_Cardinality_Inds = np.where(Minority_State_Masses >= Fixed_Feature_Permutable_Cardinality)[0]
    ##### False positive error scenarios #####
    ### Error scenario (2), SD = 1, RF < QF (Fixed Feature is RF) ###
    # Extract the group 1 and group 2 cardinalities (Fixed Feature is RF).
    Minority_Group_Cardinality = Fixed_Feature_Permutable_Cardinality.copy()
    Majority_Group_Cardinality = Sample_Cardinality - Minority_Group_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESE)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Minority_State_Masses[Higher_Minority_Cardinality_Inds])/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESE are identified from the boundaries of the ESE curve.
    Min_Entropy_ID_1 = np.zeros(Higher_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = np.repeat(Minority_Group_Cardinality,Min_Entropy_ID_1.shape[0])
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Higher_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the non-overlapping minority states when SD = 1
    Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
    Split_Direction = 1
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses[Higher_Minority_Cardinality_Inds[Sort_Into_Inds]],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
    ## Calculate divergences
    Divergences = Split_Permute_Entropies - Minimum_Entropies       
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    #EPs[EPs < 0] = 0
    Saved_EPs[0,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = EPs
    ESSs[0,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = Sort_Gains * Sort_Weights
    ### Error scenario (4), SD = -1, RF < QF (Fixed Feature is RF) ###
    # Because cluster is RF again, we can carry on variables from previous calculations and
    # continue at the point where the SD is changed to -1.
    # Num_Divergent_Cell is the overlapping minority states (Split_Permute_Value) when SD = -1 so we'll just input it as such in the DPC calculation.
    Num_Divergent_Cell = Split_Permute_Value.copy()
    Split_Direction = -1
    Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Out_Of_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses[Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]],Min_Entropy_ID_1[Sort_Out_Of_Inds],Min_Entropy_ID_2[Sort_Out_Of_Inds],Split_Permute_Value[Sort_Out_Of_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies - Minimum_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Split_Permute_Value[Sort_Out_Of_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Max_Entropy_Permutation[Sort_Out_Of_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    #EPs[EPs < 0] = 0
    Saved_EPs[1,Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] = EPs
    ESSs[1,Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] = Sort_Gains * Sort_Weights
    ##### False negative error scenarios #####
    Minority_Group_Cardinality = Fixed_Feature_Permutable_Cardinality.copy()
    Majority_Group_Cardinality = Sample_Cardinality - Minority_Group_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESE)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Minority_State_Masses[Higher_Minority_Cardinality_Inds])/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESE are identified from the boundaries of the ESE curve.
    Min_Entropy_ID_1 = np.zeros(Higher_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = np.repeat(Minority_Group_Cardinality,Higher_Minority_Cardinality_Inds.shape[0])
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Higher_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the non-overlapping minority states when SD = 1
    Num_Divergent_Cell = Minority_State_Masses[Higher_Minority_Cardinality_Inds] - Split_Permute_Value
    Split_Direction = 1
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses[Higher_Minority_Cardinality_Inds[Sort_Into_Inds]],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    #EPs[EPs < 0] = 0
    Saved_EPs[2,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = EPs
    ESSs[2,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = Sort_Gains * Sort_Weights
    # Null/Ignore points that are not applicable due to being boundary points and hence would be equal to 0.
    Saved_EPs[np.isinf(Saved_EPs)] = 0
    Saved_EPs[np.isnan(Saved_EPs)] = 0
    # Compress all of the error scenarios into a single vector.
    # Because we want to give the user the option to look at ESSs and EPs seperatley, we have to do a little bit of juggling
    # to keep the negative values while compressing the different error scenarios.
    Negative_EPs = np.where(Saved_EPs.min(axis=0) < 0)[0]
    Saved_EPs = np.max(np.absolute(Saved_EPs),axis=0)
    Saved_EPs[Negative_EPs] = Saved_EPs[Negative_EPs] * -1
    ESSs = np.max(ESSs,axis=0)
    #
    Results.append(ESSs)
    Results.append(Saved_EPs)
    return Results


### Given the constants and x values for the Entropy Sort Equation, Calculate_Fixed_RG_Sort_Values will calculate teh conditional entropy
# between the reference feature and the query feature.

def Calculate_Fixed_RG_Sort_Values(Outputs,Split_Direction,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value):
    # Calculate critical points on the ES curve
    Max_Permuation_Entropies = Calc_RG_Entropies(Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses)
    if Split_Direction == -1 and Outputs != 1:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_1,Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses)
    if Split_Direction == 1 and Outputs != 1:
        # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
        Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_2,Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses)
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_RG_Entropies(Split_Permute_Value,Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses)
    if Outputs == 1:
        return Split_Permute_Entropies, Max_Permuation_Entropies
    if Outputs == 2:
        # Calculate ES parabola properties
        Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
        Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
        # Vector of Information Gain values for each QG/RG pair.
        Sort_Gains = Entropy_Losses/Max_Entropy_Differences
        # Vector of Split Weights values for each QG/RG pair.
        Sort_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
        return Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights
    if Outputs == 3:
        # Calculate ES parabola properties
        Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
        Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
        return Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies


### Caclcuate entropies based on group cardinalities with fixed RG
def Calc_RG_Entropies(x,Group1_Cardinality,Group2_Cardinality,Minority_State_Masses):
    ## Entropy Sort Equation (ESQ) is split into for parts for convenience
    # Equation 1
    Eq_1 = np.zeros(x.shape[0])
    Calculate_Inds = np.where((x/Group1_Cardinality) > 0)[0]
    Eq_1[Calculate_Inds] = - (x[Calculate_Inds]/Group1_Cardinality)*np.log2(x[Calculate_Inds]/Group1_Cardinality)  
    # Equation 2  
    Eq_2 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group1_Cardinality - x)/Group1_Cardinality) > 0)[0]
    Eq_2[Calculate_Inds] = - ((Group1_Cardinality - x[Calculate_Inds])/Group1_Cardinality)*np.log2(((Group1_Cardinality - x[Calculate_Inds])/Group1_Cardinality))  
    # Equation 3  
    Eq_3 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Minority_State_Masses - x) / Group2_Cardinality) > 0)[0]
    Eq_3[Calculate_Inds] = - ((Minority_State_Masses[Calculate_Inds] - x[Calculate_Inds]) / Group2_Cardinality)*np.log2(((Minority_State_Masses[Calculate_Inds] - x[Calculate_Inds]) / Group2_Cardinality))
    # Equation 4  
    Eq_4 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group2_Cardinality-Minority_State_Masses+x)/Group2_Cardinality) > 0)[0]
    Eq_4[Calculate_Inds] = - ((Group2_Cardinality-Minority_State_Masses[Calculate_Inds]+x[Calculate_Inds])/Group2_Cardinality)*np.log2(((Group2_Cardinality-Minority_State_Masses[Calculate_Inds]+x[Calculate_Inds])/Group2_Cardinality))
    # Calculate overall entropy for each RG/QG pair
    Entropy = (Group1_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_1 + Eq_2) + (Group2_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_3 + Eq_4)
    return Entropy


### Whereas the previous functions focussed on calculating the ESSs and EPs, pairwise for every feature, Calculate_Individual_ESS_EPs takes
# a single feature as it's Fixed_Feature input, and outputs the ESS and EP values for that feature against every other feature in the
# scaled matrix provided in the "path" varible.

def Calculate_Individual_ESS_EPs(Fixed_Feature,path,EP_Masked_ESSs=True):
    ### Load ESFW objects
    Minority_State_Masses = np.load(path + "Minority_State_Masses.npy")
    Sample_Cardinality = np.load(path + "Sample_Cardinality.npy")
    Feature_Cardinality = np.load(path + "Feature_Cardinality.npy")
    Scaled_Matrix_Non_Zero_Inds = np.load(path + "Scaled_Matrix_Non_Zero_Inds.npy")
    Scaled_Matrix_Non_Zero_Values = np.load(path + "Scaled_Matrix_Non_Zero_Values.npy")
    ###
    #print(Feature_Ind)
    if EP_Masked_ESSs != True:
        Results = []
    Saved_EPs = np.zeros(Minority_State_Masses.shape[0])
    ESSs = np.zeros(Minority_State_Masses.shape[0])
    #
    Fixed_Feature_Permutable_Cardinality = np.sum(Fixed_Feature)
    Fixed_Feature_Non_Zero_Values = Fixed_Feature[Scaled_Matrix_Non_Zero_Inds[0]]
    Overlaps = np.minimum(Fixed_Feature_Non_Zero_Values,Scaled_Matrix_Non_Zero_Values)
    Non_Zero_Overlaps = np.where(Overlaps != 0)[0]
    Overlaps = np.bincount(Scaled_Matrix_Non_Zero_Inds[1][Non_Zero_Overlaps], Overlaps[Non_Zero_Overlaps],minlength=Feature_Cardinality)
    # Identify if the features will have smaller or larger minority state cardinalities than the cluster labels.
    Lower_Minority_Cardinality_Inds = np.where(Minority_State_Masses < Fixed_Feature_Permutable_Cardinality)[0]
    Higher_Minority_Cardinality_Inds = np.where(Minority_State_Masses >= Fixed_Feature_Permutable_Cardinality)[0]
    ##### Compare cluster labels with each feature #####
    ##### Go through all applicable False Positive scenarios #####
    ### For all FP scenarios, we follow the MaxEnt principle and force the QF to be the feature
    # with higher minority state cardinality.
    ### Error scenario (2), SD = 1, RF < QF (cluster is RF) ###
    # Extract the group 1 and group 2 cardinalities (cluster is RF).
    Minority_Group_Cardinality = Fixed_Feature_Permutable_Cardinality
    Majority_Group_Cardinality = Sample_Cardinality - Minority_Group_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESE)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Minority_State_Masses[Higher_Minority_Cardinality_Inds])/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESE are identified from the boundaries of the ESE curve.
    Min_Entropy_ID_1 = np.zeros(Higher_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = np.repeat(Minority_Group_Cardinality,Min_Entropy_ID_1.shape[0])
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Higher_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the non-overlapping minority states when SD = 1
    Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
    Split_Direction = 1
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses[Higher_Minority_Cardinality_Inds[Sort_Into_Inds]],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies - Minimum_Entropies       
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    EPs[EPs < 0] = 0
    Saved_EPs[0,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = Saved_EPs[0,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] + EPs
    ESSs[0,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = Sort_Gains * Sort_Weights
    ### Error scenario (4), SD = -1, RF < QF (cluster is RF) ###Sort_Out_Of_Inds
    # Because cluster is RF again, we can carry on variables from previous calculations and
    # continue at the point where the SD is changed to -1.
    # Num_Divergent_Cell is the overlapping minority states (Split_Permute_Value) when SD = -1 so we'll just input it as such in the DPC calculation.
    Num_Divergent_Cell = Split_Permute_Value.copy()
    Split_Direction = -1
    Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Out_Of_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses[Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]],Min_Entropy_ID_1[Sort_Out_Of_Inds],Min_Entropy_ID_2[Sort_Out_Of_Inds],Split_Permute_Value[Sort_Out_Of_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies - Minimum_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Split_Permute_Value[Sort_Out_Of_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Max_Entropy_Permutation[Sort_Out_Of_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    EPs[EPs < 0] = 0
    Saved_EPs[1,Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] = Saved_EPs[1,Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] + EPs
    ESSs[1,Higher_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] = Sort_Gains * Sort_Weights
    ### Error scenario (8), SD = -1, RF < QF (cluster is QF) ###
    # Extract the group 1 and group 2 cardinalities (cluster is QF).
    Minority_Group_Cardinality = Minority_State_Masses[Lower_Minority_Cardinality_Inds]
    Majority_Group_Cardinality = Sample_Cardinality - Minority_State_Masses[Lower_Minority_Cardinality_Inds]
    Permutable = Fixed_Feature_Permutable_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
    Min_Entropy_ID_1 = np.zeros(Lower_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = Minority_Group_Cardinality
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Lower_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the overlapping minority states (Split_Permute_Value) when SD = -1 so we'll just input it as such in the DPC calculation.
    Num_Divergent_Cell = Split_Permute_Value.copy()   
    Split_Direction = -1
    Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Sort_Out_Of_Inds],Minority_Group_Cardinality[Sort_Out_Of_Inds],Majority_Group_Cardinality[Sort_Out_Of_Inds],Min_Entropy_ID_1[Sort_Out_Of_Inds],Min_Entropy_ID_2[Sort_Out_Of_Inds],Split_Permute_Value[Sort_Out_Of_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies - Minimum_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Split_Permute_Value[Sort_Out_Of_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Max_Entropy_Permutation[Sort_Out_Of_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    EPs[EPs < 0] = 0
    Saved_EPs[2,Lower_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] = Saved_EPs[2,Lower_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] + EPs
    ESSs[2,Lower_Minority_Cardinality_Inds[Sort_Out_Of_Inds]] = Sort_Gains * Sort_Weights
    # Identify if the features will have smaller or larger minority state cardinalities than the cluster labels.
    Lower_Minority_Cardinality_Inds = np.where(Minority_State_Masses < Fixed_Feature_Permutable_Cardinality)[0]
    Higher_Minority_Cardinality_Inds = np.where(Minority_State_Masses >= Fixed_Feature_Permutable_Cardinality)[0]
    ##### FNs #####
    ##### Compare cluster labels with each feature #####
    # Identify if the features will have smaller or larger minority state cardinalities than the cluster labels.
    Lower_Minority_Cardinality_Inds = np.where(Minority_State_Masses < Fixed_Feature_Permutable_Cardinality)[0]
    Higher_Minority_Cardinality_Inds = np.where(Minority_State_Masses >= Fixed_Feature_Permutable_Cardinality)[0]
    ### Error scenario (6), SD = 1, RF < QF (cluster is QF) ###
    # Extract the group 1 and group 2 cardinalities (cluster is QF).
    Minority_Group_Cardinality = Minority_State_Masses[Lower_Minority_Cardinality_Inds]
    Majority_Group_Cardinality = Sample_Cardinality - Minority_State_Masses[Lower_Minority_Cardinality_Inds]
    Permutable = Fixed_Feature_Permutable_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
    Min_Entropy_ID_1 = np.zeros(Lower_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = Minority_Group_Cardinality
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Lower_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the non-overlapping minority states when SD = 1
    Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
    Split_Direction = 1
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality[Sort_Into_Inds],Majority_Group_Cardinality[Sort_Into_Inds],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies - Minimum_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    EPs[EPs < 0] = 0
    Saved_EPs[3,Lower_Minority_Cardinality_Inds[Sort_Into_Inds]] = Saved_EPs[3,Lower_Minority_Cardinality_Inds[Sort_Into_Inds]] + EPs
    ESSs[3,Lower_Minority_Cardinality_Inds[Sort_Into_Inds]] = Sort_Gains * Sort_Weights
    # Extract the group 1 and group 2 cardinalities (cluster is RF).
    Minority_Group_Cardinality = Fixed_Feature_Permutable_Cardinality
    Majority_Group_Cardinality = Sample_Cardinality - Minority_Group_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESE)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Minority_State_Masses[Higher_Minority_Cardinality_Inds])/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESE are identified from the boundaries of the ESE curve.
    Min_Entropy_ID_1 = np.zeros(Higher_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = np.repeat(Minority_Group_Cardinality,Higher_Minority_Cardinality_Inds.shape[0])
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Higher_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the non-overlapping minority states when SD = 1
    Num_Divergent_Cell = Minority_State_Masses[Higher_Minority_Cardinality_Inds] - Split_Permute_Value
    Split_Direction = 1
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Minority_State_Masses[Higher_Minority_Cardinality_Inds[Sort_Into_Inds]],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    EPs[EPs < 0] = 0
    Saved_EPs[4,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = Saved_EPs[4,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] + EPs
    ESSs[4,Higher_Minority_Cardinality_Inds[Sort_Into_Inds]] = Sort_Gains * Sort_Weights
    ###
    Minority_Group_Cardinality = Minority_State_Masses[Lower_Minority_Cardinality_Inds]
    Majority_Group_Cardinality = Sample_Cardinality - Minority_State_Masses[Lower_Minority_Cardinality_Inds]
    Permutable = Fixed_Feature_Permutable_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
    Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
    # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
    Min_Entropy_ID_1 = np.zeros(Lower_Minority_Cardinality_Inds.shape[0])
    Min_Entropy_ID_2 = Minority_Group_Cardinality
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Overlaps[Lower_Minority_Cardinality_Inds]
    # Num_Divergent_Cell is the overlapping minority states (Split_Permute_Value) when SD = -1 so we'll just input it as such in the DPC calculation.
    Num_Divergent_Cell = Permutable - Split_Permute_Value
    Split_Direction = 1
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality[Sort_Into_Inds],Majority_Group_Cardinality[Sort_Into_Inds],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
    ## Calculate Divergence Information
    Divergences = Split_Permute_Entropies
    # Find the average divergence for each cell that is diverging from the optimal sort.
    DPC = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
    # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
    # Max_Num_Cell_Divergences is dependent on SD
    Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
    DPC_independent = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
    # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
    EPs = DPC-DPC_independent
    EPs[EPs < 0] = 0
    Saved_EPs[5,Lower_Minority_Cardinality_Inds[Sort_Into_Inds]] = Saved_EPs[5,Lower_Minority_Cardinality_Inds[Sort_Into_Inds]] + EPs
    ESSs[5,Lower_Minority_Cardinality_Inds[Sort_Into_Inds]] = Sort_Gains * Sort_Weights
    # Combine local and global EPs
    #EPs = Saved_EPs
    # Null/Ignore points that aren't usable.
    Saved_EPs[np.isinf(Saved_EPs)] = 0
    Saved_EPs[np.isnan(Saved_EPs)] = 0
    #
    Saved_EPs = np.max(Saved_EPs,axis=0)
    ESSs = np.max(ESSs,axis=0)
    # Features with EPs > 0 provide evidence to switch expression states
    #Informative_Genes = np.where((EPs > 0))[0]
    # Get feature weights via average feature divergence
    #Average_Feature_Divergence = np.mean(EPs[Informative_Genes])
    if EP_Masked_ESSs != True:
        Results.append(ESSs)
        Results.append(Saved_EPs)
        return Results
    ESSs[Saved_EPs==0] = 0
    return ESSs


### Given the constants and x values for the Entropy Sort Equation, Calculate_Fixed_QG_Sort_Values will calculate teh conditional entropy
# between the reference feature and the query feature.

def Calculate_Fixed_QG_Sort_Values(Outputs,Split_Direction,Permutable,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value):
    # Calculate critical points on the ES curve
    Max_Permuation_Entropies = Calc_QG_Entropies(Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Split_Direction == -1 and Outputs != 1:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Minimum_Entropies = Calc_QG_Entropies(Min_Entropy_ID_1,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Split_Direction == 1 and Outputs != 1:
        # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
        Minimum_Entropies = Calc_QG_Entropies(Min_Entropy_ID_2,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_QG_Entropies(Split_Permute_Value,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Outputs == 1:
        return Split_Permute_Entropies, Max_Permuation_Entropies
    if Outputs == 2:
        # Calculate ES parabola properties
        Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
        Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
        # Vector of Information Gain values for each QG/RG pair.
        Sort_Gains = Entropy_Losses/Max_Entropy_Differences
        # Vector of Split Weights values for each QG/RG pair.
        Sort_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
        return Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies, Sort_Gains, Sort_Weights
    if Outputs == 3:
        # Calculate ES parabola properties
        Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
        Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
        return Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies


### Caclcuate entropies based on group cardinalities with fixed QG
def Calc_QG_Entropies(x,Group1_Cardinality,Group2_Cardinality,Permutable):
    ## Entropy Sort Equation (ESQ) is split into for parts for convenience
    # Equation 1
    Eq_1 = np.zeros(x.shape[0])
    Calculate_Inds = np.where((x/Group1_Cardinality) > 0)[0]
    Eq_1[Calculate_Inds] = - (x[Calculate_Inds]/Group1_Cardinality[Calculate_Inds])*np.log2(x[Calculate_Inds]/Group1_Cardinality[Calculate_Inds])  
    # Equation 2  
    Eq_2 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group1_Cardinality - x)/Group1_Cardinality) > 0)[0]
    Eq_2[Calculate_Inds] = - ((Group1_Cardinality[Calculate_Inds] - x[Calculate_Inds])/Group1_Cardinality[Calculate_Inds])*np.log2(((Group1_Cardinality[Calculate_Inds] - x[Calculate_Inds])/Group1_Cardinality[Calculate_Inds]))  
    # Equation 3  
    Eq_3 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Permutable - x) / Group2_Cardinality) > 0)[0]
    Eq_3[Calculate_Inds] = - ((Permutable - x[Calculate_Inds]) / Group2_Cardinality[Calculate_Inds])*np.log2(((Permutable - x[Calculate_Inds]) / Group2_Cardinality[Calculate_Inds]))
    # Equation 4  
    Eq_4 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group2_Cardinality-Permutable+x)/Group2_Cardinality) > 0)[0]
    Eq_4[Calculate_Inds] = - ((Group2_Cardinality[Calculate_Inds]-Permutable+x[Calculate_Inds])/Group2_Cardinality[Calculate_Inds])*np.log2(((Group2_Cardinality[Calculate_Inds]-Permutable+x[Calculate_Inds])/Group2_Cardinality[Calculate_Inds]))
    # Calculate overall entropy for each QG/RG pair
    Entropy = (Group1_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_1 + Eq_2) + (Group2_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_3 + Eq_4)
    return Entropy
