# Extract Features
Before proceeding with model inference, the following preprocessing is required. You can run the files in order, and for detailed explanations and arguments for each step, please refer to the description of each step.

### Preprocessing steps to extract features
- Note: The name given as prefix **must be the same** in 1. preprocess.py and 2. featurize.py.
#### 1. preprocess.py
* In this step, protein-ligand complexes with PDB file formats are neeeded to be processed.
* Note: The ligand lines in the PDB file should be start with "HETATM"
* Before this step, Sometimes the ligand atom type of all decoy complex files may not be the same. In this case, you can use the following files. ->  correct_ligand_atoms.py
* Expected outputs: holo.pdb, LG.params, <pdb_id>_conf.npy
```
usage: preprocess.py -i <docked_decoy_path> --prefix <save_prefix> -o <out_processed> --out-af <out_plddt> --ligand-name <ligand_name_in_pdb> --exclude <exclude.txt (option)>

e.g. python preprocess.py -i ../../example_data/gald/raw_docked/1jt1 --prefix 1jt1 -o ../../example_data//gald/processed/1jt1 --out-af ../../example_data/gald/af_plddt --ligand-name LG1 --exclude exclude.txt
```

#### 2. featurize.py
* Expected outputs: <pdb_id>.prop.npz, <pdb_id>.lig.npz
```
usage: featurize.py -i <input_path> --prefix <save_prefix> -o <save_feature_path> --exclude <exclude.txt (option)>

e.g. python3 featurize.py -i ../../example_data/gald/processed/1jt1 --prefix 1jt1 -o ../../example_data/gald/features --exclude exclude.txt
```
