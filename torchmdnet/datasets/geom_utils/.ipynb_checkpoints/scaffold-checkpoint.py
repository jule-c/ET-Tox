import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import json
from collections import defaultdict
import pandas as pd
from argparse import ArgumentParser
import os.path as osp


def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffold_splits(
             smiles_list: list,
             include_chirality: bool = False,
             frac_train: float = 0.8,
             frac_valid: float = 0.1,
             frac_test: float = 0.1,
             seed: int = 0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        scaffold = generate_scaffold(smiles, include_chirality)
        scaffolds[scaffold].append(ind)
    
    scaffold_sets = rng.permutation(np.array(list(scaffolds.values())))

    n_total_valid = int(np.floor(frac_valid * len(smiles_list)))
    n_total_test = int(np.floor(frac_test * len(smiles_list)))
    assert ((n_total_valid + n_total_test) <= int((frac_valid + frac_test)*len(smiles_list))), "Split failed"
    train_index = []
    valid_index = []
    test_index = []

    for scaffold_set in scaffold_sets:
        if len(valid_index) + len(scaffold_set) <= n_total_valid:
            valid_index.extend(scaffold_set)
        elif len(test_index) + len(scaffold_set) <= n_total_test:
            test_index.extend(scaffold_set)
        else:
            train_index.extend(scaffold_set)
    train_index, valid_index, test_index = np.array(train_index), np.array(valid_index), np.array(test_index)
    
    return train_index, valid_index, test_index



def get_argparse():
    parser = ArgumentParser(
        description="Scaffold Splitting Script for MoleculeNet dataset."
    )

    parser.add_argument("--dataset", type=str, default="tox21", choices=["tox21", "toxcast"])
    parser.add_argument("--include_chirality", default=False, action="store_true")
    parser.add_argument("--frac_train", type=float, default=0.8)
    parser.add_argument("--frac_valid", type=float, default=0.1)
    parser.add_argument("--frac_test", type=float, default=0.1)
    parser.add_argument("--summary_path", type=str, default="/workspace7/GEOM/molecule_net/raw/summary.json")

    args = parser.parse_args()

    return args


def main():
    args = get_argparse()
    smiles_list = json.loads(open(args.summary_path, "r").read())
    smiles_list = {key: val for key, val in smiles_list.items() if args.dataset in 
                   val.get('datasets', [])}

    smiles_list_ = smiles_list.copy()
    conf_per_mol = 1
    for smiles, meta_mol in tqdm(smiles_list_.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            del smiles_list[smiles]
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            del smiles_list[smiles]
            continue
        if u_conf < conf_per_mol:
            del smiles_list[smiles]
            continue
    print(f"Number of SMILES after check: {len(smiles_list)}. Should be: {len(smiles_list_)}")

    # check for uniqueness
    N = len(smiles_list)
    smiles_list = list(set(smiles_list))
    NN = len(smiles_list)
    if N != NN:
        print("Duplicate Smiles in summary")

    train_index, valid_index, test_index = generate_scaffold_splits(smiles_list=smiles_list,
                                                                    include_chirality=args.include_chirality,
                                                                    frac_train=args.frac_train,
                                                                    frac_valid=args.frac_valid,
                                                                    frac_test=args.frac_test
                                                                    )

    smiles_list = np.array(smiles_list)
    getter = lambda x: smiles_list[x]
    smiles_train, smiles_val, smiles_test = map(getter, (train_index, valid_index, test_index))
    smiles_train_df = pd.DataFrame()
    smiles_train_df["smiles"] = smiles_train
    smiles_train_df["split"] = "train"

    smiles_val_df = pd.DataFrame()
    smiles_val_df["smiles"] = smiles_val
    smiles_val_df["split"] = "val"

    smiles_test_df = pd.DataFrame()
    smiles_test_df["smiles"] = smiles_test
    smiles_test_df["split"] = "test"

    print(f"Ntrain={len(smiles_train)}, Nval={len(smiles_val)}, Ntest={len(smiles_test)}")

    smiles_dataset = pd.concat([smiles_train_df, smiles_val_df, smiles_test_df], axis=0)
    save = osp.join(osp.dirname(args.summary_path), f"{args.dataset}_split_dataset.csv")
    smiles_dataset.to_csv(save)
    
    np.savez(osp.join(osp.dirname(args.summary_path), f"{args.dataset}_split_1_geom.npz"), train_index, 
            idx_train= train_index,
            idx_val= valid_index,
            idx_test= test_index,
           )

if __name__ == '__main__':
    main()