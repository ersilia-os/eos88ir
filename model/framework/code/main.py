# imports
import sys

import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out

from cdi_predict import predict, EMB_DIM

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]


# my model: SMILES -> 8192-d Chemical Dice (CDI-Generalized) embedding
def my_model(smiles_list):
    return predict(smiles_list)


# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = my_model(smiles_list)

# check input and output have the same length
assert len(smiles_list) == outputs.shape[0]

# featurizer output columns: feat_0000 .. feat_8191
header = [f"feat_{str(i).zfill(4)}" for i in range(EMB_DIM)]

# write output in a .csv file
write_out(outputs, header, output_file, np.float32)
