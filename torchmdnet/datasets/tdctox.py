import os
import pickle
import copy
import json
from collections import defaultdict
import numpy as np
import random
import torch
from rdkit import Chem
import networkx as nx
import imp
import h5py
from tqdm import tqdm
import hashlib
import torch as pt
from glob import glob
import pandas as pd
import ase.units as units
import csv
from torch.nn.utils.rnn import pad_sequence
import re
import h5py
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data, Dataset


def smiles_to_int(smiles, vocab):
    vocab_size = len(vocab)
    return [vocab[char] for char in smiles]

def int_to_smiles(ints, vocab):
    smiles = ""
    vocab_size = len(vocab)
    vocab = {v: k for k, v in vocab.items()}
    smi = [vocab[int(i)] for i in ints]
    for s in smi:
        smiles += s
    return smiles

def valid_smiles(smi, vocab, smiles_strings):
    smiles = ""
    num_strings = torch.count_nonzero(smi)
    sm = smi[:num_strings]
    smiles = int_to_smiles(sm, vocab)
    return smiles in smiles_strings

def parse_xtb_xyz(filename, select_random=False, conf_per_mol=None):
    atom_encoder = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Al': 8, 'Si': 9,
    'P': 10, 'S': 11, 'Cl': 12, 'K': 13, 'Ca': 14, 'Ti': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Fe': 19, 'Co': 20, 'Ni': 21, 'Cu': 22, 'Zn': 23, 'Ge': 24, 'As': 25, 'Se': 26, 'Br': 27, 'Zr': 28, 'Mo': 29, 'Pd': 30, 'Ag': 31, 'Cd': 32, 'In': 33, 'Sn': 34, 'Sb': 35, 'I': 36, 'Ba': 37, 'Nd': 38, 'Gd': 39, 'Yb': 40, 'Pt': 41, 'Au': 42, 'Hg': 43, 'Pb': 44, 'Bi': 45}
    atomic_nb = [1, 3, 5,  6,  7,  8,  9, 11, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 40, 42, 46, 47, 48, 49, 50, 51, 53, 56, 60, 64, 70, 78, 79, 80, 82, 83]
    
    if conf_per_mol > 1:
        select_by_energy = True
    else:
        select_by_energy = False
    num_atoms = 0
    atom_type = []
    energy = 0
    pos = []
    with open(filename, 'r') as f:
        
        if select_random or select_by_energy:
            lines = f.readlines()
            num_atoms_total = int(lines[0])
            num_confs = len(lines) // (num_atoms_total + 2)
            
            if select_random:
                idx = random.randint(0, num_confs-1)
                random_conf = [(num_atoms_total+2)*idx, (num_atoms_total+2)*idx+num_atoms_total+2]
                lines = lines[random_conf[0]:random_conf[1]]
                #print(f"Crest conformer with id {idx} chosen")
                                
                for line_num, line in enumerate(lines):
                    if line_num == 0:
                        num_atoms = int(line)
                    elif line_num == 1:
                        # xTB outputs energy in Hartrees: Hartree to eV
                        try:
                            energy = np.array(float(line.split(" ")[-1]) * units.Hartree, dtype=np.float32)
                        except:
                            energy = np.array(float(line.split(" ")[2]) * units.Hartree, dtype=np.float32)
                    elif line_num >= 2:
                        t, x, y, z = line.split()
                        try:
                            atom_type.append(atomic_nb[atom_encoder[t]])
                        except:
                            t = t[0] + t[1].lower()
                            atom_type.append(atomic_nb[atom_encoder[t]])
                        pos.append([parse_float(x), parse_float(y), parse_float(z)])
                        
            else:
                energies = dict()
                idx = 0
                num_atoms = 0
                for line_num, line in enumerate(lines):
                    if line_num == 1 + num_atoms:
                        # xTB outputs energy in Hartrees: Hartree to eV
                        try:
                            energy = np.array(float(line.split(" ")[-1]) * units.Hartree, dtype=np.float32)
                            energies[idx] = energy
                        except:
                            energy = np.array(float(line.split(" ")[2]) * units.Hartree, dtype=np.float32)
                            energies[idx] = energy
                        idx += 1
                        num_atoms += num_atoms_total + 2
                assert len(energies) == num_confs
                
                max_confs = conf_per_mol if num_confs > conf_per_mol else num_confs
                energies = sorted(energies.items(), key=lambda x:abs(x[1]))[:max_confs]
                ids = [e[0] for e in energies]
                ranges = [[(num_atoms_total+2)*i, (num_atoms_total+2)*i+num_atoms_total+2] for i in ids]
                selected_lines = [lines[ranges[i][0]:ranges[i][1]] for i in range(len(ranges))]
                lines = selected_lines
                
                energies = []
                atom_types = []
                positions = []
                for conf in lines:
                    atom_type = []
                    pos = []
                    for line_num, line in enumerate(conf):
                        if line_num == 0:
                            num_atoms = int(line)
                        elif line_num == 1:
                            # xTB outputs energy in Hartrees: Hartree to eV
                            try:
                                energy = np.array(float(line.split(" ")[-1]) * units.Hartree, dtype=np.float32)
                            except:
                                energy = np.array(float(line.split(" ")[2]) * units.Hartree, dtype=np.float32)
                        elif line_num >= 2:
                            t, x, y, z = line.split()
                            try:
                                atom_type.append(atomic_nb[atom_encoder[t]])
                            except:
                                t = t[0] + t[1].lower()
                                atom_type.append(atomic_nb[atom_encoder[t]])
                            pos.append([parse_float(x), parse_float(y), parse_float(z)])
                            
                    assert np.array(pos, dtype=np.float32).shape[0] == num_atoms
                    assert np.array(atom_type, dtype=np.int64).shape[0] == num_atoms
                    
                    energies.append(energy)
                    atom_types.append(np.array(atom_type, dtype=np.int64))
                    positions.append(np.array(pos, dtype=np.float32))
                    
                #assert num_atoms_total == num_atoms
                result = {
                    'num_atoms': num_atoms,
                    'z': atom_types,
                    'energy': energies,
                    'pos': positions,
                }
                return result
        
        else:
            for line_num, line in enumerate(f):

                if line_num == 0:
                    num_atoms = int(line)
                elif line_num == 1:
                    # xTB outputs energy in Hartrees: Hartree to eV
                    try:
                        energy = np.array(float(line.split(" ")[-1]) * units.Hartree, dtype=np.float32)
                    except:
                        energy = np.array(float(line.split(" ")[2]) * units.Hartree, dtype=np.float32)
                elif line_num >= 2:
                    t, x, y, z = line.split()
                    try:
                        atom_type.append(atomic_nb[atom_encoder[t]])
                    except:
                        t = t[0] + t[1].lower()
                        atom_type.append(atomic_nb[atom_encoder[t]])
                    pos.append([parse_float(x), parse_float(y), parse_float(z)])
    
    #assert num_atoms_total == num_atoms
    num_atoms = num_atoms if not select_by_energy else num_atoms * conf_per_mol
    assert np.array(pos, dtype=np.float32).shape[0] == num_atoms
    assert np.array(atom_type, dtype=np.int64).shape[0] == num_atoms
    result = {
        'num_atoms': num_atoms,
        'z': np.array(atom_type, dtype=np.int64),
        'energy': energy,
        'pos': np.array(pos, dtype=np.float32),
    }
    return result

def parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        base, power = s.split('*^')
        return float(base) * 10 ** float(power)

class TDCTox(Dataset):

    """
    TDCtox dataset https://tdcommons.ai/single_pred_tasks/tox/
    """

    HARTREE_TO_EV = 27.211386246
    BORH_TO_ANGSTROM = 0.529177

    @property
    def raw_file_names(self):
        return f"{self.dataset}", f"{self.dataset}.csv"
    
    @property
    def processed_file_names(self):
        return [
            f"{self.name}.idx_{self.conformer}_{self.num_conformers}.mmap",
            f"{self.name}.z_{self.conformer}_{self.num_conformers}.mmap",
            f"{self.name}.pos_{self.conformer}_{self.num_conformers}.mmap",
            f"{self.name}.y_{self.conformer}_{self.num_conformers}.mmap",
            f"{self.name}.Q_{self.conformer}_{self.num_conformers}.mmap",
            f"{self.name}.smiles_{self.conformer}_{self.num_conformers}.mmap",
            f"{self.name}.tox_labels_{self.conformer}_{self.num_conformers}.mmap",
        ]

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        num_conformers=1,
        conformer='best',
        dataset='herg',
    ):
        self.num_conformers = num_conformers
        self.conformer = conformer
        self.dataset = dataset
        self.name = f"{self.dataset}"
        max_len_smiles = {'herg': 188, 'ames': 175, 'dili': 174, 'ld50': 174, 'skin_reaction': 98, 'carcinogen': 292, 'mutagenicity': 489}
        self.max_len_smiles = max_len_smiles[dataset]
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx_name, z_name, pos_name, y_name, Q_name, smiles_name, tox_labels_name = self.processed_paths
        
        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.y_mm = np.memmap(y_name, mode="r", dtype=np.float64)
        self.Q_mm = np.memmap(Q_name, mode="r", dtype=np.float32)
        self.smiles_mm = np.memmap(smiles_name, mode="r", dtype=np.int64, shape=(self.y_mm.shape[0], self.max_len_smiles))
        self.tox_labels_mm = np.memmap(
            tox_labels_name, mode="r", dtype=np.float32, shape=(self.y_mm.shape[0], 1)
        )

        
        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1
        assert len(self.tox_labels_mm) == len(self.y_mm)

    def sample_iter(self):

        assert len(self.raw_paths) == 2

        xyz_path = self.raw_paths[0]
        
        if self.num_conformers == 1:
            conformer = "crest_best_conformer" if self.conformer == 'best' else 'xtb_conformer' if self.conformer == 'xtb' else 'crest_conformers'
            xyz_files = glob(os.path.join(xyz_path, f"{conformer}/*.xyz"))
            xyz_files.sort(key=lambda f: int(re.sub('\D', '', f)))
            if self.conformer == 'best' or self.conformer == 'random':
                mol_indices = [int(file.split("/")[-1].split("_")[0]) for file in xyz_files]
            else:
                mol_indices = [int(file.split("/")[-1].split("_")[-1].split(".")[0]) for file in xyz_files]
        else:
            xyz_files = glob(os.path.join(xyz_path, "crest_conformers/*.xyz"))
            xyz_files.sort(key=lambda f: int(re.sub('\D', '', f)))
            mol_indices = [int(file.split("/")[-1].split("_")[0]) for file in xyz_files]

        csv_path = self.raw_paths[1]
        csv_file = pd.read_csv(csv_path)
            
        
        smiles_strings = list(csv_file['Drug']) if self.dataset != 'mutagenicity' else list(csv_file['Canonical_Smiles'])
        unique_chars = sorted(list(set([char for smiles in set(smiles_strings) for char in smiles])))
        vocab = {char: i+1 for i, char in enumerate(unique_chars)}
        vocab['<pad>'] = 0
        smiles_ints = [smiles_to_int(i, vocab) for i in smiles_strings]
        smiles_ints = pad_sequence([torch.tensor(i) for i in smiles_ints], batch_first=True).long()
        
        j = 0
        for i, d in tqdm(csv_file.iterrows()):
            if i in mol_indices:
                if self.dataset != 'mutagenicity':
                    tox_label = d['Y']
                    smiles = smiles_ints[i]
                    mol = Chem.MolFromSmiles(d['Drug'])
                elif self.dataset == 'mutagenicity':
                    tox_label = d['Activity']
                    smiles = smiles_ints[i]
                    try:
                        mol = Chem.MolFromSmiles(d['Canonical_Smiles'])
                    except:
                        continue
                else:
                    raise Exception("Dataset not found!")

                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    mol = Chem.AddHs(mol)
                    Q = Chem.GetFormalCharge(mol)
                except:
                    continue
                    
                if not valid_smiles(smiles, vocab, smiles_strings):
                    raise(f"\nNot a valid SMILES strings at index {i}...")

                file = xyz_files[j]
                
                if self.num_conformers > 1:
                    assert self.conformer == 'best'
                    
                mol_dict = parse_xtb_xyz(file, self.conformer=='random', self.num_conformers)
                j += 1
                    
                if self.num_conformers > 1:
                    for k in range(len(mol_dict['z'])):
                        labels = {
                            'tox_labels': torch.tensor(tox_label, dtype=torch.float32),
                            'Q': torch.tensor(Q, dtype=torch.float32),
                            'smiles': smiles,
                            'y': torch.tensor(mol_dict['energy'][k]),
                            'pos': torch.tensor(mol_dict['pos'][k], dtype=torch.float32),
                            'z' : torch.tensor(mol_dict['z'][k], dtype=torch.int64),
                        } 
                        yield Data(**labels)

                else: 
                    labels = {
                        'tox_labels': torch.tensor(tox_label, dtype=torch.float32),
                        'Q': torch.tensor(Q, dtype=torch.float32),
                        'smiles': smiles,
                        'y': torch.tensor(mol_dict['energy']),
                        'pos': torch.tensor(mol_dict['pos'], dtype=torch.float32),
                        'z' : torch.tensor(mol_dict['z'], dtype=torch.int64),
                    }
                    yield Data(**labels)
            elif i not in mol_indices and not self.dataset == 'mutagenicity' and self.num_conformers == 1:
                labels = {
                    'tox_labels': torch.tensor(tox_label, dtype=torch.float32),
                    'Q': torch.tensor(Q, dtype=torch.float32),
                    'smiles': smiles,
                    'y': torch.zeros((1, 1), dtype=torch.float32),
                    'pos': torch.zeros((1, 3), dtype=torch.float32),
                    'z' : torch.zeros((1, 1), dtype=torch.float32)
                }
                yield Data(**labels)
            
            else:
                continue
            
    def process(self):

        print("Arguments")
        print("Gathering statistics...")
        num_all_confs = 0
        num_all_atoms = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]

        print(f"  Total number of conformers: {num_all_confs}")
        print(f"  Total number of atoms: {num_all_atoms}")

        idx_name, z_name, pos_name, y_name, Q_name, smiles_name, tox_labels_name = self.processed_paths
        idx_mm = np.memmap(
            idx_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs + 1,)
        )
        z_mm = np.memmap(
            z_name + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
        )
        pos_mm = np.memmap(
            pos_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        y_mm = np.memmap(
            y_name + ".tmp", mode="w+", dtype=np.float64, shape=(num_all_confs,)
        )
        Q_mm = np.memmap(
            Q_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_confs,)
        )
        smiles_mm = np.memmap(
            smiles_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs, self.max_len_smiles)
        )
        tox_labels_mm = np.memmap(
            tox_labels_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_confs, 1)
        )

        print("Storing data...")
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            pos_mm[i_atom:i_next_atom] = data.pos
            y_mm[i_conf] = data.y
            Q_mm[i_conf] = data.Q
            smiles_mm[i_conf, :] = data.smiles
            tox_labels_mm[i_conf, :] = data.tox_labels

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        Q_mm.flush()
        smiles_mm.flush()
        tox_labels_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(Q_mm.filename, Q_name)
        os.rename(smiles_mm.filename, smiles_name)
        os.rename(tox_labels_mm.filename, tox_labels_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):
        
        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(
            1, 1
        )  # It would be better to use float64, but the trainer complaints
        Q = pt.tensor(self.Q_mm[idx], dtype=pt.float32).view(
            1, 1
        )
        smiles = pt.tensor(self.smiles_mm[idx], dtype=pt.int64).unsqueeze(0)
        tox_labels = pt.tensor(self.tox_labels_mm[idx], dtype=pt.float32).unsqueeze(0)

        return Data(z=z, pos=pos, y=y, Q=Q, smiles=smiles, tox_labels=tox_labels)



