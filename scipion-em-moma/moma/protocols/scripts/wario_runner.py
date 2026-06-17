import os
import sys
import time
import shutil
import argparse
import warnings
import itertools
import MDAnalysis
import multiprocessing
import umap
import numba
import pynndescent
import seaborn as sns
import pandas as pd
import ot
import h5py

import numpy as np
import h5py
import hdbscan
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial


from Bio import PDB
from tqdm import tqdm
from ipywidgets import widgets

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



def contact_features(ensemble_path, ensemble_name, thresholds, interactive = True, N_cores = 1, start = None, end = None, subsequence = None, sel_chain = None, umap_dim = 10):
            
    # Initial parameters
    var_dict = {'multiframe' : 'n', 'check_folder' : True, 'do_xtc' : False, 'do_pdb' : False,
                'N' : 1, 'start' : start, 'end' : end, 'subsequence' : subsequence,
                'ensemble_name' : ensemble_name, 'ensemble_path' : ensemble_path}
    
    var_dict['xtc_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".xtc")] 
    var_dict['pdb_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".pdb") or file.endswith(".prmtop") or file.endswith(".gsd") or file.endswith(".hdf5") or file.endswith(".mol2") or file.endswith(".hoomdxml") or file.endswith(".prm7") or file.endswith(".arc") or file.endswith(".parm7") or file.endswith(".gro") or file.endswith(".pdb.gz") or file.endswith(".pdb.gz") or file.endswith(".h5") or file.endswith(".lh5") or file.endswith(".psf")]
    var_dict['folders'] = [file for file in os.listdir(ensemble_path)  if (os.path.isdir("/".join([ensemble_path,file])) and not file.startswith('.'))]
    

        
    print("\n----------------------------------------------------------------------------------\n")
    print(' \(·_·)                                                                  \(·_·)')
    print('   ) )z                        This is WARIO!                             ) )z')
    print("   / \\                                                                     / \\ \n")
    if interactive == True:
        print("Before launching the computation, let me check I understood everything correctly...")
    print("\n----------------------------------------------------------------------------------\n")
    
    # File processing
    
    print("".join(["For the ensemble named ",var_dict["ensemble_name"],', I found ',
                    str(len(var_dict["xtc_files"])),' .xtc file(s), ',str(len(var_dict["pdb_files"])),' .pdb file(s) and ',
                    str(len(var_dict["folders"])),' folder(s).']))
        
    if len(var_dict["xtc_files"]) + len(var_dict["folders"]) + len(var_dict["pdb_files"]) == 0:
        sys.exit("".join(['Folder for ', var_dict["ensemble_name"], ' ensemble is empty...']))
        
    # .xtc file with a .pdb topology file
    
    if len(var_dict["xtc_files"]) >= len(var_dict["pdb_files"]) and len(var_dict["pdb_files"]) == 1:
        
        if interactive == True:
            print('\nShould I interprete this input as:\n')
        else:
            print('\nI will interprete this input as:\n')
        print("".join([str(var_dict["xtc_files"][0]),' : trajectory of ',var_dict["ensemble_name"],',']))
        print("".join([str(var_dict["pdb_files"][0]),' : topology file of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["xtc_files"]) > 1:
            print("\nMore than one .xtc file were found. Taking the first as the trajectory file.\n")
        if interactive == True:
            ens_input = input('...? (y/n)')
        else:
            ens_input = 'y'
        if ens_input == 'n':
            var_dict['multiframe'] = input("Should I ignore .xtc files and consider the .pdb file as a multiframe file? (y/n)")
        else:
            var_dict["do_xtc"] = True
            var_dict["xtc_root_path"] = var_dict["ensemble_path"]
            var_dict['check_folder'] = False
                
    # multiframe .pdb files

    if var_dict['multiframe'] == 'y' or (len(var_dict["pdb_files"]) >= 1 and len(var_dict["xtc_files"]) == 0):
        
        if interactive == True:
            print('\nShould I interprete this input as:\n')
        else:
            print('\nI will interprete this input as:\n')   
        print("".join([str(var_dict["pdb_files"][0]),' : trajectory of ',var_dict["ensemble_name"],'.']))
        
        if len(var_dict["pdb_files"]) > 1:
            print("\nMore than one multiframe .pdb file were found. Taking the first as the trajectory file.\n")
        if interactive == True:
            ens_input = input('...? (y/n)')
        else: 
            ens_input = 'y'
                    
        if ens_input == 'y':
            print('Trajectory has been given as multiframe .pdb file, which is not supported.')
            print("Converting file to .xtc + topology .pdb...\n ")
            if not os.path.exists("/".join([var_dict["ensemble_path"],'converted_files'])):
                os.mkdir("/".join([var_dict["ensemble_path"],'converted_files']))
            multiframe_pdb_to_xtc(pdb_file = "/".join([var_dict["ensemble_path"],var_dict["pdb_files"][0]]), save_path = "/".join([var_dict["ensemble_path"],'converted_files']), prot_name = var_dict["pdb_files"][0].split('.pdb')[0])
            print("Done.")
            var_dict["do_xtc"] = True
            var_dict["xtc_root_path"] = "/".join([var_dict["ensemble_path"],'converted_files'])
            var_dict["xtc_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".xtc")]
            var_dict["pdb_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".pdb")]
            var_dict['check_folder'] = False
                
    # folder with .pdb files
    
    if len(var_dict["folders"]) >= 1 and var_dict['check_folder'] == True:
        
        if interactive:
            print('\nShould I interprete this input as:\n')
        else:
            print('\nI will interprete this input as:\n')
        print("".join([var_dict["folders"][0],' folder contains: trajectory of ',var_dict["ensemble_name"],"."]))
        
        if len(var_dict["folders"]) > 1:
            print("\nMore than one .pdb folder were found. Taking the first as the trajectory folder.\n")
        
        if interactive:
            ens_input = input('...? (y/n)')
        else:
            ens_input = 'y'
        if ens_input == 'y':
            var_dict["do_pdb"] = True
    
    if not var_dict["do_pdb"] and not var_dict["do_xtc"]:
        sys.exit("".join(['\n Sorry, I did not understood the input. Please follow the guidelines described in the function documentation to create ',ensemble_name,' folder.\n']))    
            
    print("\n----------------------------------------------------------------------------------\n")
                    
    if interactive == True:
        print("Everything seems OK!\n")
        print("".join(['There are ',str(os.cpu_count()),' threads (cores) available.']))
        n_cores = int(input("Please specify the number of threads (cores) you would like to use (positive integer):"))
    else:
        if N_cores == 'max':
            n_cores = int(os.cpu_count())
        else:
            n_cores = int(N_cores)
    
    if not os.path.exists("/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])):
        os.mkdir("/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])]))
        results_path = "/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])
    if os.path.exists("/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])):
        if len(os.listdir("/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])]))) == 0:
            results_path = "/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])
        else:
            sys.exit("".join(['The folder ', "/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])]),' already exists and it is not empty. Please empty or delete it.']))   
    
    print("\n----------------------------------------------------------------------------------\n")
    print("3..."); time.sleep(1); 
    print("2..."); time.sleep(1)
    print("1..."); time.sleep(1)
    print("Go!"); time.sleep(0.2)
    print("\n----------------------------------------------------------------------------------")
    
    # Build frames and save coordinates
    
    print('\nComputing contact weights for ' + var_dict["ensemble_name"] + '...\n')
        
    if var_dict["do_xtc"] == True:
                    
        wcontact_matrix(thresholds, xtc_file = "/".join([var_dict["xtc_root_path"],var_dict["xtc_files"][0]]), top_file = "/".join([var_dict["xtc_root_path"],var_dict["pdb_files"][0]]),
                    pdb_folder = None, num_cores = n_cores, prot_name = ensemble_name, save_to =  results_path,
                    start = var_dict["start"], end = var_dict["end"], subsequence = var_dict["subsequence"],
                    select_chain = sel_chain,
                    name_variable = '__main__')
            
    if var_dict["do_pdb"] == True:
                    
        wcontact_matrix(thresholds, xtc_file = None, top_file = None, pdb_folder = "/".join([var_dict["ensemble_path"],var_dict["folders"][0]]), num_cores = n_cores,
                            prot_name = ensemble_name, save_to =  results_path,
                            start = var_dict["start"], end = var_dict["end"], subsequence = var_dict["subsequence"],
                            select_chain = sel_chain,
                            name_variable = '__main__')
    
    print("\n----------------------------------------------------------------------------------\n")
    print("Contact weights computed.\n")
    print("Embedding data into a 2-dimensional UMAP space for visualization...\n")
    
    h5f = h5py.File("/".join([results_path, "_".join([ensemble_name,'wcontmatrix.h5'])]),'r')
    wcont_data = h5f['data'][:]
    h5f.close()
    
    n_neighbors = min(15, wcont_data.shape[0] - 1)
    #n_neighbors = 15
    embedding_2d = umap.UMAP(random_state = 42,
                        n_neighbors = n_neighbors,
                        min_dist = 0.1).fit_transform(wcont_data)
    np.save('/'.join([results_path, "_".join([ensemble_name,'embedding_2d_wcont'])]), embedding_2d)

    
    print("".join(["Done! Embedding data into a ",str(umap_dim),"-dimensional UMAP space...\n"]))
    
    n_neighbors = min(30, wcont_data.shape[0] - 1)
    #n_neighbors = 30
    n_components = min(umap_dim, wcont_data.shape[1] - 1)
    #n_components = umap_dim
    clusterable_embedding = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = 0.0,
        n_components = n_components,
        random_state = 42,
        init='random'
            ).fit_transform(wcont_data)
    np.save('/'.join([results_path, "_".join([ensemble_name,'clusterable_embedding_wcont'])]), clusterable_embedding)
    
    
    print("\n----------------------------------------------------------------------------------\n")
    print("Done! Clustering can be performed in the low-dimensional space.")

def multiframe_pdb_to_xtc(pdb_file, save_path, prot_name):
    
    u = MDAnalysis.core.universe.Universe(pdb_file)
    at = u.atoms
    
    os.chdir(save_path)
    
    # Write the trajectory in .xtc format
    at.write(".".join([prot_name,'xtc']), frames='all')
    # Write a frame of the trajectory in .pdb format for topology information
    at.write(".".join([prot_name,'pdb'])) 

def wcontact_matrix(thresholds, num_cores = 1, prot_name = None, save_to = None, pdb_folder = None, xtc_file = None, top_file = None, start = None, end = None, subsequence = None, select_chain = None, name_variable = '__main__'):
    
    if save_to is None and prot_name is None:
        
        quit('Please set save_to = None or prot_name != None and save_to != None.')
    
    if xtc_file is None and top_file is None and pdb_folder is not None:
                
        traj_file = None
        conf_list = os.listdir(pdb_folder)
        N_conformations = len(conf_list) # Number of conformations
        
    elif xtc_file is not None and top_file is not None and pdb_folder is None:
        
        if top_file.endswith(".gro"):
            top_file = md.formats.GroTrajectoryFile(top_file).topology
        
        traj_file = md.load_xtc(xtc_file, top = top_file)
        N_conformations = len(traj_file)
        conf_list = np.arange(N_conformations)        
        
    else:
        quit('Please set pdb_folder != None and xtc_file = top_file = None, or pdb_folder = None and xtc_file != None, top_file != None.')
    
    
    def comp_function(conf_comp, thresholds_comp, pdb_data_comp, traj_data_comp, start_comp, end_comp, subset_comp, sel_chain):
        
        coordinates = get_coordinates(conf_name = conf_comp, pdb = pdb_data_comp, traj = traj_data_comp, res_start = start_comp, res_end = end_comp, seq_subset = subset_comp, which_chain = sel_chain)
        contacts = get_contacts(coordinates, thresholds_comp)
        return contacts
    
    it_function = partial(comp_function, thresholds_comp = thresholds, pdb_data_comp = pdb_folder, traj_data_comp = traj_file, start_comp = start, end_comp = end, subset_comp = subsequence, sel_chain = select_chain) 
    N_pairs = len(it_function(conf_list[0]))  
    
    def it_function_error(conf):
        
        try:
            output = it_function(conf)
        except:
            output = np.repeat(np.nan, N_pairs)
        return output


    #if __name__ == name_variable:
    os.environ['PYTHONWARNINGS'] = 'ignore'
    wcont_matrix = Parallel(n_jobs = num_cores, prefer = 'processes')(delayed(it_function_error)(i) for i in tqdm(conf_list))   
    wcont_data = pd.DataFrame(np.reshape(np.asarray(wcont_matrix), [len(conf_list), N_pairs]))
        
    if save_to is None:
        
        return wcont_data

    elif save_to is not None and prot_name is not None:
        
        h5f = h5py.File('_'.join(['/'.join([save_to, prot_name]), 'wcontmatrix.h5']), 'w')
        h5f.create_dataset("data", data = wcont_data)
        h5f.close()    

def get_coordinates(conf_name, pdb = None, traj = None, res_start = None, res_end = None, seq_subset = None, which_chain = None):
    
    aa_list = list(["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU","GLY","HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO", "SER", "THR", "TRP", "TYR", "VAL"])
    
    parser = PDB.PDBParser()

    def get_structure(conf_name, conf_path, chain_id = None): 
    
        os.chdir(conf_path)
        struct = parser.get_structure('prot',conf_name)

        coor_x=list()
        coor_y=list()
        coor_z=list()
        model_list=list()
        chain_list=list()
        residue_list=list()
        atom_list=list()
        position_list=list()
        
        for model in struct:
            for chain in model:                
                for residue in chain:
                    for atom in residue:
                        x,y,z = atom.get_coord()
                        coor_x.append(x)
                        coor_y.append(y)
                        coor_z.append(z)
                        model_list.append(1+model.id)
                        chain_list.append(chain.id)
                        residue_list.append(residue.get_resname())
                        atom_list.append(atom.id)
                        position_list.append(residue.get_full_id()[3][1])
                                    
        data = {'Model': model_list,
                'Chain': chain_list,
                'Residue': residue_list,
                'Atom': atom_list,
                'Position': position_list,
                'coor_x': coor_x,
                'coor_y': coor_y,
                'coor_z': coor_z
                }
                    
        df = pd.DataFrame (data, columns = ['Model','Chain','Residue','Atom','Position','coor_x','coor_y','coor_z'],index=None)
        df = df[df.Model == df.Model[0]] # Keep just one model
        if chain_id is None:
            df = df[df.Chain == df.Chain[0]] # Keep just one chain
        else:
            df = df[df.Chain == chain_id] # Keep just one chain
        
        # Check for HIE
        df = df.replace('HIE', 'HIS')
                    
        return df
            
    if traj is None and pdb is not None:

        df = get_structure(conf_name, conf_path = pdb, chain_id = which_chain)
        # Remove residues not in aa_list
        df = df.loc[df["Residue"].isin(aa_list),]
        L = len(np.unique(df.Position))
        
    elif pdb is None and traj is not None:
        
        traj = traj[conf_name]
        top_table = traj.top.to_dataframe()[0]
        df = pd.concat([top_table, pd.DataFrame(traj.xyz[0], columns = np.array(['x','y','z']))], axis = 1)
        df = df[['segmentID','chainID','resName','name','resSeq','x','y','z']]
        df.columns = ['Model','Chain','Residue','Atom','Position','coor_x','coor_y','coor_z']
        # Remove residues not in aa_list
        df = df.loc[df["Residue"].isin(aa_list),]
        L = len(np.unique(df.Position))
        
        # Correct division by 10
        df.loc[:,['coor_x', 'coor_y', 'coor_z']] = df.loc[:,['coor_x', 'coor_y', 'coor_z']]*10

    if seq_subset is not None and res_start is None and res_end is None:   
        df = df.loc[np.isin(df.Position, seq_subset)]
        
        
    if seq_subset is not None and (res_start is not None or res_end is not None):
        print('\nseq_subset has been ignored as res_start or res_end are not set to None. Set res_start and res_end to None to use seq_subset.\n')

    if res_start is None:
        
        res_start = np.min(df.Position)
    
    if res_end is None:
        
        res_end = np.max(df.Position)
        
    if seq_subset is None:
        
        seq_subset = np.arange(res_start, res_end + 1, 1)
        
    L = len(np.unique(df.Position))

    
    # Build reference systems

    basis_angles = np.array([1.917213, 1.921843, 2.493444])
    b = np.array([np.cos(basis_angles[0]), np.cos(basis_angles[1]), np.cos(basis_angles[2])]).T
    
    # 1. Definition of the reference frame on every sequence position

    CA_coor = df.loc[ (df.Atom == 'CA') , ['coor_x','coor_y','coor_z']].to_numpy() # CA coordinates 
    print(df.loc[df.Position == 6, 'Atom'].tolist())


    if np.shape(CA_coor)[0] != L:
        sys.exit('The number of CA atoms does not match the number of residues.')
            
    N_coor = df.loc[ (df.Atom == 'N')  , ['coor_x','coor_y','coor_z']].to_numpy() # N coordinates 
    
    if np.shape(N_coor)[0] != L:
        sys.exit('The number of N atoms does not match the number of residues.')
    
    C_coor = df.loc[ (df.Atom == 'C')  , ['coor_x','coor_y','coor_z']].to_numpy() # C coordinates 
    
    if np.shape(C_coor)[0] != L:
        sys.exit('The number of C atoms does not match the number of residues.')

    N_CA_coor = N_coor - CA_coor; N_CA_coor = N_CA_coor / np.linalg.norm(N_CA_coor, axis = 1)[:, None]
    C_CA_coor = C_coor - CA_coor; C_CA_coor = C_CA_coor / np.linalg.norm(C_CA_coor, axis = 1)[:, None]
    CxN_coor = np.cross(C_CA_coor, N_CA_coor); CxN_coor = CxN_coor / np.linalg.norm(CxN_coor, axis = 1)[:, None]
    
    A_list = np.concatenate([N_CA_coor,C_CA_coor,CxN_coor], axis = 1)
    A_list = np.reshape(A_list, [np.shape(A_list)[0]*3, 3])
    A_list = [A_list[i:(i+3),:] for i in 3*np.arange(np.shape(N_CA_coor)[0])]

    A_array = np.array(A_list)
    b_array = np.array([b for i in np.arange(len(A_list))])

    CB_coor = np.linalg.solve(A_array, b_array)


    # Reference frames 
    
    b1_coor = CB_coor / np.linalg.norm(CB_coor, axis = 1)[:, None] # b1 = CA-CB
    CN_coor = N_CA_coor - C_CA_coor # CN
    b2_coor = np.cross(CN_coor, b1_coor); b2_coor = b2_coor / np.linalg.norm(b2_coor, axis = 1)[:, None] # b2 = b1 x CN
    b3_coor = np.cross(b1_coor, b2_coor); b3_coor = b3_coor / np.linalg.norm(b3_coor, axis = 1)[:, None] # b3 = b1 x b2 = CN for a perfect tetrahedron
    
    P_list = np.concatenate([b1_coor, b2_coor, b3_coor], axis = 1)
    P_list = np.reshape(P_list, [np.shape(P_list)[0]*3, 3]).T
    P_list = [P_list[:,i:(i+3)] for i in 3*np.arange(np.shape(b1_coor)[0])]
    P_list = np.linalg.inv(np.array(P_list)) # Change-of-basis matrix for each position
    
    positions = df.loc[ ((df.Atom =='CB') & (df.Residue!='GLY')) | ((df.Atom =='CA') & (df.Residue=='GLY')), ['coor_x','coor_y','coor_z']]

    pos_pairs = np.array(list(itertools.combinations(range(L), 2)))
    P_list_pairs = [P_list[i] for i in pos_pairs[:,0]]
    positions_pairs = positions.to_numpy()[pos_pairs[:,1],:] - positions.to_numpy()[pos_pairs[:,0],:]
    or1_pairs = b1_coor[pos_pairs[:,1],:]
    or2_pairs = b3_coor[pos_pairs[:,1],:]
    
    relative_pairwise_positions = np.einsum('ij,ikj->ik',positions_pairs, P_list_pairs)
    relative_pairwise_or1 = np.einsum('ij,ikj->ik', or1_pairs, P_list_pairs)
    relative_pairwise_or2 = np.einsum('ij,ikj->ik', or2_pairs, P_list_pairs)
        
    aa_seq = df.Residue[df.Atom == 'CA'].to_numpy()
    d = {item: idx for idx, item in enumerate(aa_list)}
    aa_index = np.array([d.get(item) for item in aa_seq])
    aa_pairs = np.concatenate([aa_index[pos_pairs[:,0]][:,None],aa_index[pos_pairs[:,1]][:, None]], axis = 1)
    pos_pairs = np.concatenate([seq_subset[pos_pairs[:,0]][:,None],seq_subset[pos_pairs[:,1]][:, None]], axis = 1)

    positions_and_frames = np.concatenate([relative_pairwise_positions, relative_pairwise_or1,
                                        relative_pairwise_or2, aa_pairs, pos_pairs], axis = 1)        
    
    return positions_and_frames  

def get_contacts(coor_conf, threshold_file, assort = False):

    L = int(0.5*(1 + np.sqrt(1 + 8*np.shape(coor_conf)[0])))

    contact_thresholds =  pd.read_csv(threshold_file, sep=" ", header=0)
    contact_thresholds['th11'] = np.deg2rad(contact_thresholds['th11'])
    contact_thresholds['th12'] = np.deg2rad(contact_thresholds['th12'])
    contact_thresholds['th13'] = np.deg2rad(contact_thresholds['th13'])
    contact_thresholds['th21'] = np.deg2rad(contact_thresholds['th21'])
    contact_thresholds['th22'] = np.deg2rad(contact_thresholds['th22'])
    contact_thresholds['th23'] = np.deg2rad(contact_thresholds['th23'])

    add = pd.DataFrame(contact_thresholds)
    add.columns = contact_thresholds.columns
    add = add.loc[(add.AA1 - add.AA2 != 0)]
    add[['AA1','AA2']] = add[['AA2','AA1']].values
    contact_thresholds = pd.concat([contact_thresholds,add], ignore_index = True)
    contact_thresholds['AA1'] = contact_thresholds.AA1.astype('int')
    contact_thresholds['AA2'] = contact_thresholds.AA2.astype('int')
    contact_thresholds['range'] = contact_thresholds.range.astype('int')
    contact_thresholds['AA1-AA2-range'] = contact_thresholds.AA1.astype(str) + '-' + contact_thresholds.AA2.astype(str) + '-' + contact_thresholds.range.astype(str)
    contact_thresholds = contact_thresholds[['AA1-AA2-range','delta_min','delta_max','delta', 'th11', 'th12', 'th13', 'th21', 'th22', 'th23', 'delta_se3_min', 'delta_se3_max']]
    mins = np.minimum(contact_thresholds.delta_min.values,contact_thresholds.delta_max.values)
    maxs = np.maximum(contact_thresholds.delta_min.values,contact_thresholds.delta_max.values)
    contact_thresholds.delta_min = mins
    contact_thresholds.delta_max = maxs

    coor_conf = pd.DataFrame(coor_conf,
                            columns = ['coor_x','coor_y','coor_z','or1_x','or1_y','or1_z','or2_x','or2_y','or2_z','AA1','AA2','pos1','pos2'])
    coor_conf.range = np.abs(coor_conf['pos1'] - coor_conf['pos2'])
    coor_conf.range = (coor_conf.range*(coor_conf.range<5) + 5*(coor_conf.range>=5)).astype('int') 
    coor_conf['AA1'] = coor_conf.AA1.astype('int')
    coor_conf['AA2'] = coor_conf.AA2.astype('int')
    coor_conf['AA1-AA2-range'] = coor_conf.AA1.astype(str) + '-' + coor_conf.AA2.astype(str) + '-' + coor_conf.range.astype(str)
    
    #coor_conf = coor_conf.join(vaex.from_pandas(contact_thresholds), left_on = 'AA1-AA2-range', right_on = 'AA1-AA2-range', how = 'left')
    coor_conf = coor_conf.merge(contact_thresholds, on = 'AA1-AA2-range', how = 'left')
            
    # For range <= 4, we correct distance by admissible orientations
    coor_conf['min_th1'] = np.nanmin([np.abs(np.arccos(coor_conf['or1_x']) - coor_conf['th11']),
                                                np.abs(np.arccos(coor_conf['or1_x']) - coor_conf['th12']), 
                                                np.abs(np.arccos(coor_conf['or1_x']) - coor_conf['th13'])], axis = 0)
    coor_conf['min_th2'] = np.nanmin([np.abs(np.arccos(coor_conf['or2_z']) - coor_conf['th21']),
                                                np.abs(np.arccos(coor_conf['or2_z']) - coor_conf['th22']), 
                                                np.abs(np.arccos(coor_conf['or2_z']) - coor_conf['th23'])], axis = 0) 
    coor_conf['dis_th1'] = 0.5*(np.sin(coor_conf.min_th1)**2*(coor_conf.min_th1 < np.pi/2) + (1 - np.cos(coor_conf.min_th1)**2)*(coor_conf.min_th1 >= np.pi/2))
    coor_conf['dis_th2'] = 0.5*(np.sin(coor_conf.min_th2)**2*(coor_conf.min_th2 < np.pi/2) + (1 - np.cos(coor_conf.min_th2)**2)*(coor_conf.min_th2 >= np.pi/2))
    
    alpha = beta = 0.5
    coor_conf['dis_r3'] = np.sqrt(coor_conf.coor_x**2 + coor_conf.coor_y**2 + coor_conf.coor_z**2)
    coor_conf['dis_or'] = np.sqrt(alpha*coor_conf['dis_th1']**2 + beta*coor_conf['dis_th1']**2)
    
    argtanh = lambda x: 0.5*np.log((1+x)/(1-x))
    coor_conf[coor_conf.delta_min < 2].delta_min = 2  
    coor_conf[coor_conf.delta_max <= 2].delta_max = 3 
    coor_conf['d'] = np.log(argtanh(1/coor_conf.delta_min))/np.log(coor_conf.delta_min/coor_conf.delta_max)
    coor_conf['w_or_pos'] = 1-np.tanh((coor_conf.dis_r3/coor_conf.delta_max)**coor_conf.d)
    coor_conf['a'] = 0.5*np.sqrt(argtanh(1-coor_conf.w_or_pos))
    coor_conf['w_or_or'] = 1-np.tanh((2*(coor_conf.dis_or+coor_conf.a))**2)
    coor_conf[coor_conf.w_or_pos == 0].w_or_or = 0
    coor_conf['dis_or'] = coor_conf['dis_or'].fillna(0)
    coor_conf['w_or_or'] = coor_conf['w_or_or'].fillna(0)
    coor_conf['dis_se3'] = np.sqrt((1-coor_conf.w_or_or)**2*coor_conf.dis_r3**2 + coor_conf.w_or_or**2*coor_conf.dis_or**2)  
    
    coor_conf.delta_se3_min[coor_conf['delta_se3_min'].isnull()] = coor_conf[coor_conf['delta_se3_min'].isnull()]['delta_min']
    coor_conf.delta_se3_min[coor_conf['delta_se3_min'] < 2] = 2  
    coor_conf.delta_se3_max[coor_conf['delta_se3_max'].isnull()] = coor_conf[coor_conf['delta_se3_max'].isnull()]['delta_max']
    coor_conf.delta_se3_max[coor_conf['delta_se3_max'] <= 2] = 3  
    coor_conf['d_se3'] = np.log(argtanh(1/coor_conf.delta_se3_min))/np.log(coor_conf.delta_se3_min/coor_conf.delta_se3_max)
    coor_conf['w_dis_se3'] = 1-np.tanh((coor_conf.dis_se3/coor_conf.delta_se3_max)**coor_conf.d_se3)
    
    coor_conf = coor_conf[['pos1','pos2','w_dis_se3','AA1', 'AA2']]
    coor_conf['pos1'] = coor_conf.pos1.astype('int') + 1
    coor_conf['pos2'] = coor_conf.pos2.astype('int') + 1   
    
    if assort:
        return coor_conf
    else:
        return coor_conf.w_dis_se3  
    



def plot_2umap_script(embedding_2d, labels_umap, ensemble_name, extra_path, dpi_png=200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    classified   = np.where(labels_umap >= 0)[0]
    unclassified = np.where(labels_umap < 0)[0]

    fig, ax = plt.subplots()
    ax.scatter(embedding_2d[unclassified, 0], embedding_2d[unclassified, 1],
               color=(0.5, 0.5, 0.5), s=0.5, alpha=0.5)
    ax.scatter(embedding_2d[classified, 0], embedding_2d[classified, 1],
               c=labels_umap[classified], s=0.5, alpha=1, cmap='Spectral')
    plt.xlabel('UMAP coordinate 1')
    plt.ylabel('UMAP coordinate 2')
    plt.title(f'UMAP 2D projection for {ensemble_name}', fontsize=8)
    plt.savefig(os.path.join(extra_path, f'clusters_2d_{ensemble_name}.png'), dpi=dpi_png)
    plt.close()
    print(f">>> Plot saved to {extra_path}/clusters_2d_{ensemble_name}.png")

    # Tabla de ocupación
    repartition = pd.Series(labels_umap).value_counts()
    repartition.index = ["Unclassified" if i == -1 else i for i in repartition.index]
    df_rep = pd.DataFrame({
        "Cluster": np.array(repartition.index),
        "Occupancy (%)": 100 * np.array(repartition.values) / len(labels_umap)
    })
    print(df_rep)
    df_rep.to_csv(os.path.join(extra_path, f'cluster_occupancy_{ensemble_name}.csv'), index=False)

def get_cluster_files(ensemble_path, ensemble_name, labels_umap):
             
    # Initial parameters
    var_dict = {'multiframe' : 'n', 'check_folder' : True, 'do_xtc' : False, 'do_pdb' : False,
                'ensemble_name' : ensemble_name, 'ensemble_path' : ensemble_path}
    
    var_dict['xtc_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".xtc")] 
    var_dict['pdb_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".pdb") or file.endswith(".prmtop") or file.endswith(".parm7") or file.endswith(".gro")]
    var_dict['folders'] = [file for file in os.listdir(ensemble_path)  if (os.path.isdir("/".join([ensemble_path,file])) and not file.startswith('.') and not file.startswith("results"))]
       
   # File processing
       
    if len(var_dict["xtc_files"]) + len(var_dict["folders"]) + len(var_dict["pdb_files"]) == 0:
        sys.exit("".join(['Folder for ', var_dict["ensemble_name"], ' ensemble is empty...']))
        
    # .xtc file with a .pdb topology file
    
    if len(var_dict["xtc_files"]) >= len(var_dict["pdb_files"]) and len(var_dict["pdb_files"]) == 1:

        print('\nTaking as input:\n')
        print("".join([str(var_dict["xtc_files"][0]),' : trajectory of ',var_dict["ensemble_name"],',']))
        print("".join([str(var_dict["pdb_files"][0]),' : topology file of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["xtc_files"]) > 1:
            print("\nMore than one .xtc file were found. Taking the first as the trajectory file.\n")
        var_dict["do_xtc"] = True
        var_dict["xtc_root_path"] = var_dict["ensemble_path"]
        var_dict['check_folder'] = False
                
    # multiframe .pdb files
   
    if var_dict['multiframe'] == 'y' or (len(var_dict["pdb_files"]) >= 1 and len(var_dict["xtc_files"]) == 0):
        
        print('\nTaking as input:\n')   
        print("".join([str(var_dict["pdb_files"][0]),' : trajectory of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["pdb_files"]) > 1:
            print("\nMore than one multiframe .pdb file were found. Taking the first as the trajectory file.\n")
        print("\nTaking the previously converted files.\n")
        var_dict["do_xtc"] = True
        var_dict["xtc_root_path"] = "/".join([var_dict["ensemble_path"],'converted_files'])
        var_dict["xtc_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".xtc")]
        var_dict["pdb_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".pdb")]
        var_dict['check_folder'] = False
                
    # folder with .pdb files
     
    if len(var_dict["folders"]) >= 1 and var_dict['check_folder'] == True:
        
        print('\nTaking as input:\n')
        print("".join([var_dict["folders"][0],' folder contains: trajectory of ',var_dict["ensemble_name"],"."]))
        if len(var_dict["folders"]) > 1:
            print("\nMore than one .pdb folder were found. Taking the first as the trajectory folder.\n")
        var_dict["do_pdb"] = True
    
    if not var_dict["do_pdb"] and not var_dict["do_xtc"]:
        sys.exit("".join(['\n Sorry, I did not understood the input. Please follow the guidelines described in the function documentation to create ',ensemble_name,' folder.\n']))    
            
    print("\n----------------------------------------------------------------------------------\n")
    print("\nCreating cluster-specific files...\n")
    
    results_path = "/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])
    save_files = "/".join([results_path, "cluster_files"])
    if not os.path.exists(save_files):
        os.mkdir(save_files)
    
    if var_dict["do_xtc"]:
        
        traj_file = md.load_xtc("/".join([var_dict["xtc_root_path"],var_dict["xtc_files"][0]]), top = "/".join([var_dict["xtc_root_path"],var_dict["pdb_files"][0]]))
             
        # Save .xtc cluster files
        for k in tqdm(range(len(np.unique(labels_umap[labels_umap >= 0])))):
            traj_file[np.where(labels_umap == k)].save_xtc("/".join([save_files, "".join([ensemble_name,'_',str(k),'.xtc'])]))

    if var_dict["do_pdb"]:
        
        conf_list = os.listdir("/".join([var_dict["ensemble_path"],var_dict["folders"][0]]))

        for k in tqdm(range(len(np.unique(labels_umap[labels_umap >= 0])))):
            clus_k_path = "/".join([save_files, "_".join(['clus',str(k)])])
            if not os.path.exists(clus_k_path):
                os.mkdir(clus_k_path)
            
            clus_k = np.where(labels_umap == k)[0]
            for j in range(len(clus_k)):
                traj = md.load_pdb("/".join(["/".join([var_dict["ensemble_path"],var_dict["folders"][0]]),conf_list[clus_k[j]]]))
                traj.save_pdb("/".join([clus_k_path, "".join([ensemble_name,'_',str(conf_list[clus_k[j]]),'.pdb'])]))

    print("\nFiles saved.\n")     

def get_wmaps(labels_umap, ensemble_name, results_path, subsequence=None,
              marks=None, pdf=False, dpi_png=500, fontsize_title=10,
              fontsize_axis=5, fontsize_suptitle=12, shrink_cbar=.5,
              xticks_angle=90, extra_path=None):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Guardar plots en extra_path si se proporciona, si no en results_path
    save_path = extra_path if extra_path is not None else results_path
    maps_path = os.path.join(save_path, 'wcont_maps')
    if not os.path.exists(maps_path):
        os.mkdir(maps_path)

    # El .h5 siempre está en results_path (tmp/)
    h5f = h5py.File(os.path.join(results_path, f'{ensemble_name}_wcontmatrix.h5'), 'r')
    n, p = np.shape(h5f['data'])

    L = int(0.5 * (1 + np.sqrt(1 + 8 * p)))
    print(f">>> DEBUG: The contact matrix has {p} columns, corresponding to {L} sequence positions.")
    list_pos = np.asarray(list(itertools.combinations(range(1, L + 1), 2)))

    repartition = pd.Series(labels_umap).value_counts()
    repartition.index = ["Unclassified" if i == -1 else i for i in repartition.index]
    if "Unclassified" in repartition.index:
        repartition = repartition.drop("Unclassified")

    cont_data = np.zeros([len(repartition), len(list_pos) + 1])

    for cluster in tqdm(repartition.index):

        # Datos del cluster
        cluster_data = h5f['data'][labels_umap == cluster, ]

        # Matriz de contactos del cluster
        prop_cluster = pd.Series(labels_umap).value_counts().sort_index()[cluster] / n
        cont_matrix = pd.DataFrame(
            np.concatenate([list_pos, np.asarray([cluster_data.mean(axis=0)]).T], axis=1),
            columns=['pos1', 'pos2', 'cp']
        )
        cont_matrix.pos1 = cont_matrix.pos1.astype(int)
        cont_matrix.pos2 = cont_matrix.pos2.astype(int)
        cont_data[np.where(repartition.index == cluster)[0][0], :] = np.append(
            cont_matrix.cp.to_numpy(), prop_cluster
        )
        cont_matrix = cont_matrix.pivot(index='pos1', columns='pos2', values='cp')

        if subsequence is not None:
            list_pos_idx = list_pos - 1
            cont_matrix.index   = np.unique(subsequence[list_pos_idx[:, 0]]).astype(int)
            cont_matrix.columns = np.unique(subsequence[list_pos_idx[:, 1]]).astype(int)

        fig = plt.figure(rasterized=True)
        res = sns.heatmap(cont_matrix.T, cmap='Reds', square=True,
                          cbar_kws={"shrink": shrink_cbar, 'label': "Contact weight average"})
        plt.suptitle(f"{ensemble_name} contact-based clustering", fontsize=fontsize_suptitle)
        plt.title(f"Cluster #{cluster} with {round(100 * prop_cluster, 2)}% of occupation",
                  fontsize=fontsize_title)
        plt.xlabel('Sequence position')
        plt.ylabel('Sequence position')
        plt.xticks(rotation=xticks_angle)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=fontsize_axis)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=fontsize_axis)

        if marks is not None:
            res.hlines(marks, *res.get_xlim())
            res.vlines(marks, *res.get_ylim())

        if pdf:
            plt.savefig(os.path.join(maps_path, f"{ensemble_name}_{cluster}.pdf"))
        else:
            plt.savefig(os.path.join(maps_path, f"{ensemble_name}_{cluster}.png"), dpi=dpi_png)
        plt.close()

    h5f.close()
    np.save(os.path.join(save_path, f'{ensemble_name}_contdata'), cont_data)
    print(f">>> Maps saved to {maps_path}")

def representative_ensemble(size, ensemble_path, ensemble_name, labels_umap):
             
    # Initial parameters
    var_dict = {'multiframe' : 'n', 'check_folder' : True, 'do_xtc' : False, 'do_pdb' : False,
                'ensemble_name' : ensemble_name, 'ensemble_path' : ensemble_path}
    
    var_dict['xtc_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".xtc")] 
    var_dict['pdb_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".pdb") or file.endswith(".prmtop") or file.endswith(".parm7") or file.endswith(".gro")]
    var_dict['folders'] = [file for file in os.listdir(ensemble_path)  if (os.path.isdir("/".join([ensemble_path,file])) and not file.startswith('.') and not file.startswith("results"))]
       
   # File processing
       
    if len(var_dict["xtc_files"]) + len(var_dict["folders"]) + len(var_dict["pdb_files"]) == 0:
        sys.exit("".join(['Folder for ', var_dict["ensemble_name"], ' ensemble is empty...']))
        
    # .xtc file with a .pdb topology file
    
    if len(var_dict["xtc_files"]) >= len(var_dict["pdb_files"]) and len(var_dict["pdb_files"]) == 1:

        print('\nTaking as input:\n')
        print("".join([str(var_dict["xtc_files"][0]),' : trajectory of ',var_dict["ensemble_name"],',']))
        print("".join([str(var_dict["pdb_files"][0]),' : topology file of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["xtc_files"]) > 1:
            print("\nMore than one .xtc file were found. Taking the first as the trajectory file.\n")
        var_dict["do_xtc"] = True
        var_dict["xtc_root_path"] = var_dict["ensemble_path"]
        var_dict['check_folder'] = False
                
    # multiframe .pdb files
   
    if var_dict['multiframe'] == 'y' or (len(var_dict["pdb_files"]) >= 1 and len(var_dict["xtc_files"]) == 0):
        
        print('\nTaking as input:\n')   
        print("".join([str(var_dict["pdb_files"][0]),' : trajectory of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["pdb_files"]) > 1:
            print("\nMore than one multiframe .pdb file were found. Taking the first as the trajectory file.\n")
        print("\nTaking the previously converted files.\n")
        var_dict["do_xtc"] = True
        var_dict["xtc_root_path"] = "/".join([var_dict["ensemble_path"],'converted_files'])
        var_dict["xtc_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".xtc")]
        var_dict["pdb_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".pdb")]
        var_dict['check_folder'] = False
                
    # folder with .pdb files
     
    if len(var_dict["folders"]) >= 1 and var_dict['check_folder'] == True:
        
        print('\nTaking as input:\n')
        print("".join([var_dict["folders"][0],' folder contains: trajectory of ',var_dict["ensemble_name"],"."]))
        if len(var_dict["folders"]) > 1:
            print("\nMore than one .pdb folder were found. Taking the first as the trajectory folder.\n")
        var_dict["do_pdb"] = True
    
    if not var_dict["do_pdb"] and not var_dict["do_xtc"]:
        sys.exit("".join(['\n Sorry, I did not understood the input. Please follow the guidelines described in the function documentation to create ',ensemble_name,' folder.\n']))    
            
    print("\n----------------------------------------------------------------------------------\n")
    print("\nSampling representative family...\n")
    
    repartition = pd.Series(labels_umap).value_counts() # Clustering partition
    repartition.index = ["Unclassified" if i == -1 else i for i in repartition.index]
    repartition = repartition.drop("Unclassified")
    probas = repartition.values/np.sum(repartition.values)

    selected_conf = np.zeros(size)
    for i in range(size):

        choose_cluster = np.random.choice(repartition.index, size = 1, p = probas)[0]
        selected_conf[i] = np.random.choice(np.where(labels_umap == choose_cluster)[0], size = 1)[0]
    
    selected_conf = np.ndarray.astype(selected_conf, int)
    results_path = "/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])
    save_files = "/".join([results_path, "representative_family"])
    if not os.path.exists(save_files):
        os.mkdir(save_files)
    
    if var_dict["do_xtc"]:
        
        traj_file = md.load_xtc("/".join([var_dict["xtc_root_path"],var_dict["xtc_files"][0]]), top = "/".join([var_dict["xtc_root_path"],var_dict["pdb_files"][0]]))
             
        # Save .xtc file
        traj_file[selected_conf].save_xtc("/".join([save_files, "".join([ensemble_name,'_repfam.xtc'])]))

    if var_dict["do_pdb"]:
        
        conf_list = os.listdir("/".join([var_dict["ensemble_path"],var_dict["folders"][0]]))
        # Save pdb folder
        for j in selected_conf:
            traj = md.load_pdb("/".join(["/".join([var_dict["ensemble_path"],var_dict["folders"][0]]),conf_list[j]]))
            traj.save_pdb("/".join([save_files, "".join([ensemble_name,'_',conf_list[j],'.pdb'])]))

    print("\nFiles saved.\n")     

def cluster_descriptors(ensemble_path, ensemble_name, labels_umap, subsequence = None,
                       fig_width = 10, fig_height = 1.7, shrink_cbar = .7, yticks_angle = 0,
                       fontsize_title = 8, fontsize_axis = 7, pdf = False, dpi_png = 200):
             
    # Initial parameters
    var_dict = {'multiframe' : 'n', 'check_folder' : True, 'do_xtc' : False, 'do_pdb' : False,
                'ensemble_name' : ensemble_name, 'ensemble_path' : ensemble_path}
    
    var_dict['xtc_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".xtc")] 
    var_dict['pdb_files'] = [file for file in os.listdir(ensemble_path)  if file.endswith(".pdb") or file.endswith(".prmtop") or file.endswith(".parm7") or file.endswith(".gro")]
    var_dict['folders'] = [file for file in os.listdir(ensemble_path)  if (os.path.isdir("/".join([ensemble_path,file])) and not file.startswith('.') and not file.startswith("results"))]
       
   # File processing
       
    if len(var_dict["xtc_files"]) + len(var_dict["folders"]) + len(var_dict["pdb_files"]) == 0:
        sys.exit("".join(['Folder for ', var_dict["ensemble_name"], ' ensemble is empty...']))
        
    # .xtc file with a .pdb topology file
    
    if len(var_dict["xtc_files"]) >= len(var_dict["pdb_files"]) and len(var_dict["pdb_files"]) == 1:

        print('\nTaking as input:\n')
        print("".join([str(var_dict["xtc_files"][0]),' : trajectory of ',var_dict["ensemble_name"],',']))
        print("".join([str(var_dict["pdb_files"][0]),' : topology file of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["xtc_files"]) > 1:
            print("\nMore than one .xtc file were found. Taking the first as the trajectory file.\n")
        var_dict["do_xtc"] = True
        var_dict["xtc_root_path"] = var_dict["ensemble_path"]
        var_dict['check_folder'] = False
                
    # multiframe .pdb files
   
    if var_dict['multiframe'] == 'y' or (len(var_dict["pdb_files"]) >= 1 and len(var_dict["xtc_files"]) == 0):
        
        print('\nTaking as input:\n')   
        print("".join([str(var_dict["pdb_files"][0]),' : trajectory of ',var_dict["ensemble_name"],'.']))
        if len(var_dict["pdb_files"]) > 1:
            print("\nMore than one multiframe .pdb file were found. Taking the first as the trajectory file.\n")
        print("\nTaking the previously converted files.\n")
        var_dict["do_xtc"] = True
        var_dict["xtc_root_path"] = "/".join([var_dict["ensemble_path"],'converted_files'])
        var_dict["xtc_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".xtc")]
        var_dict["pdb_files"] = [file for file in os.listdir(var_dict["xtc_root_path"]) if file.endswith(".pdb")]
        var_dict['check_folder'] = False
                
    # folder with .pdb files
     
    if len(var_dict["folders"]) >= 1 and var_dict['check_folder'] == True:
        
        print('\nTaking as input:\n')
        print("".join([var_dict["folders"][0],' folder contains: trajectory of ',var_dict["ensemble_name"],"."]))
        if len(var_dict["folders"]) > 1:
            print("\nMore than one .pdb folder were found. Taking the first as the trajectory folder.\n")
        var_dict["do_pdb"] = True
    
    if not var_dict["do_pdb"] and not var_dict["do_xtc"]:
        sys.exit("".join(['\n Sorry, I did not understood the input. Please follow the guidelines described in the function documentation to create ',ensemble_name,' folder.\n']))    
            
    print("\n----------------------------------------------------------------------------------\n")
    print("\nComputing cluster-specific descriptors...\n")
    
    results_path = "/".join([os.path.abspath(ensemble_path),"_".join(['results',ensemble_name])])
    save_files = "/".join([results_path, "cluster_descriptors"])
    if not os.path.exists(save_files):
        os.mkdir(save_files)
    
    if var_dict["do_xtc"]:
        
        traj_file = md.load_xtc("/".join([var_dict["xtc_root_path"],var_dict["xtc_files"][0]]), top = "/".join([var_dict["xtc_root_path"],var_dict["pdb_files"][0]]))
        L = traj_file.n_residues
        Nconf = traj_file.n_frames
         
        dssp_types = ['H','B','E','G','I','T','S',' ']
        prop_dssp = np.zeros([len(dssp_types),L,len(labels_umap)-1])
        rg = np.zeros([len(labels_umap)-1])
    
        for k in range(len(np.unique(labels_umap[labels_umap >= 0]))):
        
            prop_dssp_k = np.zeros([len(dssp_types),L])
            dssp_k = md.compute_dssp(traj_file[np.where(labels_umap == k)], simplified = False)
            rg[k] = np.mean(md.compute_rg(traj_file[np.where(labels_umap == k)]))
            for dt in range(len(dssp_types)):
                prop_dssp_k[dt,:] = (dssp_k == dssp_types[dt]).sum(axis = 0)/len(np.where(labels_umap == k)[0])
            prop_dssp[:,:,k] = prop_dssp_k

    if var_dict["do_pdb"]:
        
        pdb_folder = "/".join([var_dict["ensemble_path"],var_dict["folders"][0]])
        conf_list = os.listdir(pdb_folder)
        md_file = md.load_pdb("/".join([pdb_folder,conf_list[0]]))
        L = md_file.topology.n_residues
        Nconf = len(conf_list)
        
        dssp_types = ['H','B','E','G','I','T','S',' ']
        prop_dssp = np.zeros([len(dssp_types),L,len(labels_umap)-1])
        rg = np.zeros([len(labels_umap)-1])
        
        for k in range(len(np.unique(labels_umap[labels_umap >= 0]))):
            
            prop_dssp_k = np.zeros([len(dssp_types),L])
            clus_k = np.where(labels_umap == k)[0]
            dssp_k = np.zeros([len(clus_k),L]).astype(str)
            rg_k = np.zeros([len(clus_k)])

            for l in range(len(clus_k)):
                dssp_k[l,:] = md.compute_dssp(md.load_pdb("/".join([pdb_folder,conf_list[clus_k[l]]])), simplified = False)[0].astype(str)
                rg_k[l] = md.compute_rg(md.load_pdb("/".join([pdb_folder,conf_list[clus_k[l]]])))
            rg[k] = np.mean(rg_k)
            for dt in range(len(dssp_types)):
                prop_dssp_k[dt,:] = (dssp_k == dssp_types[dt]).sum(axis = 0)/len(np.where(labels_umap == k)[0])
            prop_dssp[:,:,k] = prop_dssp_k
  
    
    if subsequence is None:
    
        subsequence = np.arange(L)
    
    for cluster in tqdm(range(len(np.unique(labels_umap[labels_umap >= 0])))):
        
        prop_cluster = round(100*len(np.where(labels_umap == cluster)[0])/Nconf,2)
        fig = plt.figure(figsize=(fig_width, fig_height))
        res = sns.heatmap(prop_dssp[:,subsequence,cluster], cmap='Blues', square = True,  cbar_kws={"shrink": shrink_cbar,'label':"Class prop."})
        xlabels = [item.get_text() for item in res.get_xmajorticklabels()]
        plt.xlabel('Sequence position')
        plt.ylabel('DSSP class')
        plt.title("".join([ensemble_name, ' - cluster #',str(cluster),' (',str(prop_cluster),'% oc.). Average RG = ', str(round(10*rg[cluster],2)),r'$\AA$.']), fontsize = fontsize_title)
        plt.yticks(rotation = yticks_angle) 
        res.set_xticklabels(np.asarray(xlabels).astype(int) + 1, fontsize = fontsize_axis)
        res.set_yticklabels(['L' if x==' ' else x for x in dssp_types], fontsize = fontsize_axis)
        if pdf:
            plt.savefig("/".join([save_files,"".join([ensemble_name,'_',str(cluster),'_DSSP.pdf'])]), bbox_inches='tight')
        else:
            plt.savefig("/".join([save_files,"".join([ensemble_name,'_',str(cluster),'_DSSP.png'])]), dpi = dpi_png, bbox_inches='tight')

    print("\nPlots saved.\n")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_name', required=True)
    parser.add_argument('--ensemble_path', required=True)
    parser.add_argument('--thresholds', required=True)   # path to th_file
    parser.add_argument('--subsequence', default=None)
    parser.add_argument('--N_cores', default=1)
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--extra_path', required=True)
    args = parser.parse_args()

    # argparse passes literal strings; allow passing the special value "None"
    if args.subsequence == 'None':
        args.subsequence = None
    else:
        args.subsequence = np.array(list(map(int, args.subsequence.split(','))), dtype=int)


    contact_features(
        ensemble_name=args.ensemble_name,
        ensemble_path=args.ensemble_path,
        N_cores=int(args.N_cores),
        interactive=False,
        thresholds=args.thresholds,
        subsequence=args.subsequence
    )

    results_path = "/".join([os.path.abspath(args.ensemble_path),
                                "_".join(['results', args.ensemble_name])])
    
    embedding_2d = np.load('/'.join([results_path, "_".join([args.ensemble_name,'embedding_2d_wcont.npy'])]))
    clusterable_embedding = np.load('/'.join([results_path, "_".join([args.ensemble_name,'clusterable_embedding_wcont.npy'])]))  
        
    min_cluster_size = max(2, int(clusterable_embedding.shape[0] * 0.01))
    #min_cluster_size = int(clusterable_embedding.shape[0]*0.01)
    
    labels_umap = hdbscan.HDBSCAN(
        min_samples = 10,
        min_cluster_size = min_cluster_size, 
    ).fit_predict(clusterable_embedding)

    np.savetxt(os.path.join(args.extra_path, f'{args.ensemble_name}_labels_umap.txt'), labels_umap, fmt='%d')
    #print(f">>> DEBUG: Labels guardados en: {args.extra_path}/{args.ensemble_name}_labels_umap.txt")


    classified = np.where(labels_umap >= 0)[0]
    #print("".join(["\nThe clustering algorithm found ",str(len(np.unique(labels_umap[labels_umap >= 0])))," clusters and classified the ",str(np.round(100*len(classified)/len(labels_umap),2)),"% of conformations. \n"])) 

    centroids = {}
    h5_path = os.path.join(results_path, f"{args.ensemble_name}_wcontmatrix.h5")
    with h5py.File(h5_path, 'r') as f:
        contact_matrix = f['data'][:]

    for cluster_id in np.unique(labels_umap[labels_umap >= 0]):

        idx = np.where(labels_umap == cluster_id)[0]

        cluster_points = contact_matrix[idx] 

        baricenter = cluster_points.mean(axis=0)

        distances = np.linalg.norm(cluster_points - baricenter, axis=1)
        medoid_local_idx = np.argmin(distances)
        medoid_global_idx = idx[medoid_local_idx]

        centroids[cluster_id] = medoid_global_idx
        print(f"Cluster {cluster_id}: conformación representativa = frame {medoid_global_idx} "
            f"(Distance to the baricentre: {distances[medoid_local_idx]:.4f})")


    if os.path.isdir(os.path.join(args.ensemble_path, 'ensemble')):
        ensemble_dir = os.path.join(args.ensemble_path, 'ensemble')
    else:
        ensemble_dir = args.ensemble_path


    conformations_dir = os.path.join(ensemble_dir, 'conformations')
    if os.path.isdir(conformations_dir):
        ensemble_dir = conformations_dir

    xtc_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.xtc')]
    pdb_files  = sorted([f for f in os.listdir(ensemble_dir) if f.endswith('.pdb')])

    #print(f">>> ensemble_dir: {ensemble_dir}")
    #print(f">>> contents: {os.listdir(ensemble_dir)}")

    if xtc_files:
        traj = md.load_xtc(
            os.path.join(ensemble_dir, xtc_files[0]),
            top=os.path.join(ensemble_dir, pdb_files[0])
        )
    else:
        traj = md.load(
            [os.path.join(ensemble_dir, f) for f in pdb_files]
        )


    centroids_path = os.path.join(args.extra_path, 'centroids')
    os.makedirs(centroids_path, exist_ok=True)

    for cluster_id, frame_idx in centroids.items():
        pdb_out = os.path.join(centroids_path, f'centroid_cluster_{cluster_id}.pdb')
        traj[int(frame_idx)].save_pdb(pdb_out)

    plot_2umap_script(embedding_2d, labels_umap, args.ensemble_name, args.extra_path)
    get_wmaps(labels_umap, args.ensemble_name, results_path, args.subsequence, args.extra_path)

    results_dest = os.path.join(args.extra_path, 'results_' + args.ensemble_name)
    if os.path.exists(results_dest):
        shutil.rmtree(results_dest)
    shutil.copytree(results_path, results_dest)

