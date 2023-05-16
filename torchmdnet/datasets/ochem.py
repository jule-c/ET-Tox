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

import os
import numpy as np


def parse_xtb_xyz(filename):
    atom_encoder = {
        "H": 0,
        "Li": 1,
        "B": 2,
        "C": 3,
        "N": 4,
        "O": 5,
        "F": 6,
        "Na": 7,
        "Al": 8,
        "Si": 9,
        "P": 10,
        "S": 11,
        "Cl": 12,
        "K": 13,
        "Ca": 14,
        "Ti": 15,
        "V": 16,
        "Cr": 17,
        "Mn": 18,
        "Fe": 19,
        "Co": 20,
        "Ni": 21,
        "Cu": 22,
        "Zn": 23,
        "Ge": 24,
        "As": 25,
        "Se": 26,
        "Br": 27,
        "Zr": 28,
        "Mo": 29,
        "Pd": 30,
        "Ag": 31,
        "Cd": 32,
        "In": 33,
        "Sn": 34,
        "Sb": 35,
        "I": 36,
        "Ba": 37,
        "Nd": 38,
        "Gd": 39,
        "Yb": 40,
        "Pt": 41,
        "Au": 42,
        "Hg": 43,
        "Pb": 44,
        "Bi": 45,
    }
    atomic_nb = [
        1,
        3,
        5,
        6,
        7,
        8,
        9,
        11,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        32,
        33,
        34,
        35,
        40,
        42,
        46,
        47,
        48,
        49,
        50,
        51,
        53,
        56,
        60,
        64,
        70,
        78,
        79,
        80,
        82,
        83,
    ]

    num_atoms = 0
    atom_type = []
    pos = []
    with open(filename, "r") as f:
        for line_num, line in enumerate(f):
            if line_num < 3:
                continue
            elif line_num == 3:
                line = line.split(" ")[1]
                num_atoms = int(line)
            elif line_num > num_atoms + 3:
                break
            elif line_num > 3:
                line = np.array([f for f in line.split(" ") if f != ""])
                x, y, z = [float(line[i]) for i in range(3)]
                t = line[3]
                try:
                    atom_type.append(atomic_nb[atom_encoder[t]])
                except:
                    try:
                        t = t[0] + t[1].lower()
                        atom_type.append(atomic_nb[atom_encoder[t]])
                    except:
                        raise ValueError(
                            "Atom types in the data did not match with atom-type lookup table. Please add atom type and atom number to the lookup!"
                        )
                pos.append([parse_float(x), parse_float(y), parse_float(z)])

    # assert num_atoms_total == num_atoms
    assert np.array(pos, dtype=np.float32).shape[0] == num_atoms
    assert np.array(atom_type, dtype=np.int64).shape[0] == num_atoms
    result = {
        "num_atoms": num_atoms,
        "z": np.array(atom_type, dtype=np.int64),
        "pos": np.array(pos, dtype=np.float32),
    }
    return result


def parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        base, power = s.split("*^")
        return float(base) * 10 ** float(power)


class OChem(Dataset):

    """
    OChem dataloader (currently only for a specific data input format)
    """

    @property
    def raw_file_names(self):
        return f"{self.dataset}"

    @property
    def processed_file_names(self):
        return [
            f"{self.name}.idx.mmap",
            f"{self.name}.z.mmap",
            f"{self.name}.pos.mmap",
            f"{self.name}.tox_labels.mmap",
        ]

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset="ochem",
    ):
        self.dataset = dataset

        super().__init__(root, transform, pre_transform, pre_filter)

        (
            idx_name,
            z_name,
            pos_name,
            tox_labels_name,
        ) = self.processed_paths

        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.tox_labels_mm = np.memmap(tox_labels_name, mode="r", dtype=np.float32)

        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)

    def sample_iter(self):
        assert len(self.raw_paths) == 1

        data_path = self.raw_paths[0]
        all_files = glob(os.path.join(data_path), "*")

        for i, file in tqdm(all_files):
            mol_dict = parse_xtb_xyz(
                file, self.conformer == "random", self.num_conformers
            )
            labels = {
                "tox_labels": torch.tensor(tox_label, dtype=torch.float32),
                "pos": torch.tensor(mol_dict["pos"], dtype=torch.float32),
                "z": torch.tensor(mol_dict["z"], dtype=torch.int64),
            }
            yield Data(**labels)

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

        (
            idx_name,
            z_name,
            pos_name,
            y_name,
            Q_name,
            smiles_name,
            tox_labels_name,
        ) = self.processed_paths
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
            smiles_name + ".tmp",
            mode="w+",
            dtype=np.int64,
            shape=(num_all_confs, self.max_len_smiles),
        )
        tox_labels_mm = np.memmap(
            tox_labels_name + ".tmp",
            mode="w+",
            dtype=np.float32,
            shape=(num_all_confs, 1),
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
        Q = pt.tensor(self.Q_mm[idx], dtype=pt.float32).view(1, 1)
        smiles = pt.tensor(self.smiles_mm[idx], dtype=pt.int64).unsqueeze(0)
        tox_labels = pt.tensor(self.tox_labels_mm[idx], dtype=pt.float32).unsqueeze(0)

        return Data(z=z, pos=pos, y=y, Q=Q, smiles=smiles, tox_labels=tox_labels)
