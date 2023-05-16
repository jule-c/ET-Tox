import os
import pickle
import copy
import json
from collections import defaultdict
import numpy as np
import random
import torch
from tqdm import tqdm
import hashlib
import imp
import torch as pt
from .geom_utils.geom_utils import *
from torch.nn.utils.rnn import pad_sequence
import re
import networkx as nx
import random
from torch_geometric.data import Data, Dataset


from mendeleev import element

def electron_embedding(atomic_numbers):
    
    node_feats = []
    
    for atom in atomic_numbers:
        xi = []
        xi += torch.tensor([atom], dtype=torch.float32)

        atom = element(int(atom))
        xi += torch.tensor([atom.ec.unpaired_electrons()], dtype=torch.float32)
        xi += torch.tensor([atom.electronegativity()], dtype=torch.float32)
        xi += torch.tensor([atom.hardness()], dtype=torch.float32)
        xi += torch.tensor([atom.softness()], dtype=torch.float32)
        xi += torch.tensor([atom.electron_affinity], dtype=torch.float32)
        xi += torch.tensor([atom.atomic_radius], dtype=torch.float32)
        xi += torch.tensor([atom.atomic_volume], dtype=torch.float32)
        xi += torch.tensor([atom.density], dtype=torch.float32)
        xi += torch.tensor([atom.dipole_polarizability], dtype=torch.float32)
        xi = torch.stack(xi)
        node_feats.append(xi)
        
    node_feats = torch.stack(node_feats)
    return node_feats

def smiles_to_int(smiles, vocab):
    vocab_size = len(vocab)
    return [vocab[char] for char in smiles]

class MoleculeNetProps(Dataset):

    """
    GEOM dataset (https://github.com/learningmatter-mit/geom)
    """

    HARTREE_TO_EV = 27.211386246
    BORH_TO_ANGSTROM = 0.529177

    @property
    def raw_file_names(self):
        return "summary_old.json"
    
    @property
    def processed_file_names(self):
        return [
            f"{self.dataset}.idx_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.z_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.atom_props_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.pos_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.y_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.smiles_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.Q_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
            f"{self.dataset}.tox_labels_{self.num_conformers}_{self.data_version}_{self.conformer}.mmap",
        ]

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        num_conformers=1,
        conformer='best',
        data_version='full',
        dataset='tox21'
    ):
        self.conformer = conformer
        self.num_conformers = num_conformers
        self.data_version = data_version
        self.name = f"{self.__class__.__name__}"
        self.dataset = dataset
        max_len_smiles = {'herg': 188, 'ames': 175, 'dili': 174, 'ld50': 174, 'skin_reaction': 98, 'carcinogens': 292, 'tox21': 325, 'toxcast': 325, 'sider': 413, 'bbbp': 257, 'clintox': 323, 'bace': 194}
        num_labels = {'tox21': 12, 'toxcast': 617, 'sider': 27, 'bbbp': 1, 'clintox': 2, 'bace': 1}
        self.max_len_smiles = max_len_smiles[dataset]
        self.num_labels = num_labels[dataset]
        
        super().__init__(root, transform, pre_transform, pre_filter)
    
        idx_name, z_name, atom_props_name, pos_name, y_name, smiles_name, Q_name, tox_labels_name = self.processed_paths
        self.idx_mm = np.memmap(idx_name, mode="r", dtype=np.int64)
        self.z_mm = np.memmap(z_name, mode="r", dtype=np.int8)
        self.atom_props_mm = np.memmap(atom_props_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 10))
        self.pos_mm = np.memmap(
            pos_name, mode="r", dtype=np.float32, shape=(self.z_mm.shape[0], 3)
        )
        self.y_mm = np.memmap(y_name, mode="r", dtype=np.float64)
        self.smiles_mm = np.memmap(smiles_name, mode="r", dtype=np.int64, shape=(self.y_mm.shape[0], self.max_len_smiles))
        self.Q_mm = np.memmap(Q_name, mode="r", dtype=np.float32)
        self.tox_labels_mm = np.memmap(
            tox_labels_name, mode="r", dtype=np.float32, shape=(self.y_mm.shape[0], self.num_labels)
        )
        
        assert self.idx_mm[0] == 0
        assert self.idx_mm[-1] == len(self.z_mm)
        assert len(self.idx_mm) == len(self.y_mm) + 1
        assert len(self.tox_labels_mm) == len(self.y_mm)

    def sample_iter(self):

        assert len(self.raw_paths) == 1
        
        dataset_name = self.dataset
        conf_per_mol = self.num_conformers

        summary_path = self.raw_paths[0]
        with open(summary_path, 'r') as f:
            summ = json.load(f)
            summ = {key: val for key, val in summ.items() if dataset_name in val.get('datasets', [])}

        # filter valid pickle path
        smiles_list = []
        pickle_path_list = []
        num_mols = 0    
        num_confs = 0    
        for smiles, meta_mol in tqdm(summ.items()):
            u_conf = meta_mol.get('uniqueconfs')
            if u_conf is None:
                continue
            pickle_path = meta_mol.get('pickle_path')
            if pickle_path is None:
                continue
            #if u_conf < conf_per_mol:
            #    continue
            num_mols += 1
            num_confs += conf_per_mol
            smiles_list.append(smiles)
            pickle_path_list.append(pickle_path)
                
        smiles_strings = smiles_list
        unique_chars = sorted(list(set([char for smiles in set(smiles_strings) for char in smiles])))
        vocab = {char: i+1 for i, char in enumerate(unique_chars)}
        vocab['<pad>'] = 0
        smiles_strings = [smiles_to_int(i, vocab) for i in smiles_strings]
        smiles_strings = pad_sequence([torch.tensor(i) for i in smiles_strings], batch_first=True).long()
        
        bad_case = 0
        for i in tqdm(range(len(pickle_path_list))):
            with open(os.path.join(self.root, pickle_path_list[i]), 'rb') as fin:
                mol = pickle.load(fin)

                if mol.get('uniqueconfs') > len(mol.get('conformers')):
                    bad_case += 1
                    continue
                if mol.get('uniqueconfs') <= 0:
                    bad_case += 1
                    continue

            datas = []
            smiles = mol.get('smiles')
            
            if mol.get('uniqueconfs') == conf_per_mol:
                # use all confs
                conf_ids = np.arange(mol.get('uniqueconfs'))
            else:
                if self.conformer == "random":
                    conf_ids = [random.choice(np.arange(mol.get('uniqueconfs')))]
                else:
                # filter the most probable 'conf_per_mol' confs
                    all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
                    descend_conf_id = (-all_weights).argsort()
                    conf_ids = descend_conf_id[:conf_per_mol]

            for conf_id in conf_ids:
                conf_meta = mol.get('conformers')[conf_id]
                data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
                
                if self.dataset == "tox21":
                    tox_labels = [
                                mol['tox21']['nr-ar'],
                                mol['tox21']['nr-ar-lbd'],
                                mol['tox21']['nr-ahr'],
                                mol['tox21']['nr-aromatase'],
                                mol['tox21']['nr-er'],
                                mol['tox21']['nr-er-lbd'],
                                mol['tox21']['nr-ppar-gamma'],
                                mol['tox21']['sr-are'],
                                mol['tox21']['sr-atad5'],
                                mol['tox21']['sr-hse'],
                                mol['tox21']['sr-mmp'],
                                mol['tox21']['sr-p53']
                    ]
                    tox_labels = torch.tensor(
                        [i if i != '' else float(-100) for i in tox_labels], 
                        dtype=torch.float32)
                    assert (len(tox_labels) == 12), "There should be 12 toxicity labels for Tox21"

                        
                elif self.dataset == "sider":
                    tox_labels = [mol['sider'][k] for k in sider_dict]
                    tox_labels = torch.tensor(
                        [i if i != '' else float(-100) for i in tox_labels], 
                        dtype=torch.float32)
                    assert (len(tox_labels) == 27), "There should be 27 toxicity labels for Sider"
                    
                elif self.dataset == "clintox":
                    tox_labels = [
                                mol['clintox']['ct_tox'],
                                mol['clintox']['fda_approved'],
                    ]
                    tox_labels = torch.tensor(
                        [i if i != '' else float(-100) for i in tox_labels], 
                        dtype=torch.float32)
                    assert (len(tox_labels) == 2), "There should be 2 toxicity labels for ClinTox"
                    
                elif self.dataset == "bbbp":
                    tox_labels = [mol[dataset_name]['p_np']]
                    tox_labels = torch.tensor(
                        [i if i != '' else float(-100) for i in tox_labels], 
                        dtype=torch.float32)
                    assert (len(tox_labels) == 1), "There should be 1 toxicity label for BBBP"
                
                elif self.dataset == "bace":
                    tox_labels = [mol[dataset_name]['class']]
                    tox_labels = torch.tensor(
                        [i if i != '' else float(-100) for i in tox_labels], 
                        dtype=torch.float32)
                    assert (len(tox_labels) == 1), "There should be 1 toxicity label for BACE"
                    
                
                elif self.dataset == "toxcast":
                    tox_labels = [mol['toxcast'][k] for k in toxcast_dict]
                    tox_labels = torch.tensor(
                        [i if i != '' else float(-100) for i in tox_labels], 
                        dtype=torch.float32)
                    assert (len(tox_labels) == 617), "There should be 617 toxicity labels for ToxCast"
                    
                else:
                    raise Exception("Dataset not found!")
                
                
                data['atom_props'] = electron_embedding(data['z'])
                assert data['atom_props'].shape[0] == data['z'].shape[0]
                assert data['atom_props'].shape[1] == 10
                assert len(data['atom_props'].shape) == 2
                
                labels = {
                    'y': conf_meta['totalenergy'],
                    'smiles': smiles_strings[i],
                    'boltzmannweight': conf_meta['boltzmannweight'],
                    'Q': mol['charge'],
                    'tox_labels': tox_labels,
                }
                for k, v in labels.items():
                    if k == 'tox_labels':
                        data[k] = v.unsqueeze(0)
                    elif k == 'smiles':
                        data[k] = v
                    else:
                        data[k] = torch.tensor([v], dtype=torch.float32)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                yield data

    def process(self):

        print("Arguments")
        print("Gathering statistics...")
        num_all_confs = 0
        num_all_atoms = 0
        for data in self.sample_iter():
            num_all_confs += 1
            num_all_atoms += data.z.shape[0]
        
        self.num_all_atoms = num_all_atoms
        print(f"  Total number of conformers: {num_all_confs}")
        print(f"  Total number of atoms: {num_all_atoms}")

        idx_name, z_name, atom_props_name, pos_name, y_name, smiles_name, Q_name, tox_labels_name = self.processed_paths
        idx_mm = np.memmap(
            idx_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs + 1,)
        )
        z_mm = np.memmap(
            z_name + ".tmp", mode="w+", dtype=np.int8, shape=(num_all_atoms,)
        )
        atom_props_mm = np.memmap(
            atom_props_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 10)
        )
        pos_mm = np.memmap(
            pos_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_atoms, 3)
        )
        y_mm = np.memmap(
            y_name + ".tmp", mode="w+", dtype=np.float64, shape=(num_all_confs,)
        )
        smiles_mm = np.memmap(
            smiles_name + ".tmp", mode="w+", dtype=np.int64, shape=(num_all_confs, self.max_len_smiles)
        )
        Q_mm = np.memmap(
            Q_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_confs,)
        )
        tox_labels_mm = np.memmap(
            tox_labels_name + ".tmp", mode="w+", dtype=np.float32, shape=(num_all_confs, self.num_labels)
        )

        print("Storing data...")
        i_atom = 0
        for i_conf, data in enumerate(self.sample_iter()):
            i_next_atom = i_atom + data.z.shape[0]

            idx_mm[i_conf] = i_atom
            z_mm[i_atom:i_next_atom] = data.z.to(pt.int8)
            atom_props_mm[i_atom:i_next_atom] = data.atom_props.to(pt.float32)
            pos_mm[i_atom:i_next_atom] = data.pos
            y_mm[i_conf] = data.y
            smiles_mm[i_conf, :] = data.smiles
            Q_mm[i_conf] = data.Q
            tox_labels_mm[i_conf, :] = data.tox_labels

            i_atom = i_next_atom

        idx_mm[-1] = num_all_atoms
        assert i_atom == num_all_atoms

        idx_mm.flush()
        z_mm.flush()
        atom_props_mm.flush()
        pos_mm.flush()
        y_mm.flush()
        smiles_mm.flush()
        Q_mm.flush()
        tox_labels_mm.flush()

        os.rename(idx_mm.filename, idx_name)
        os.rename(z_mm.filename, z_name)
        os.rename(atom_props_mm.filename, atom_props_name)
        os.rename(pos_mm.filename, pos_name)
        os.rename(y_mm.filename, y_name)
        os.rename(smiles_mm.filename, smiles_name)
        os.rename(Q_mm.filename, Q_name)
        os.rename(tox_labels_mm.filename, tox_labels_name)

    def len(self):
        return len(self.y_mm)

    def get(self, idx):

        atoms = slice(self.idx_mm[idx], self.idx_mm[idx + 1])
        z = pt.tensor(self.z_mm[atoms], dtype=pt.long)
        atom_props = pt.tensor(self.atom_props_mm[atoms], dtype=pt.float32)
        pos = pt.tensor(self.pos_mm[atoms], dtype=pt.float32)
        y = pt.tensor(self.y_mm[idx], dtype=pt.float32).view(
            1, 1
        )  # It would be better to use float64, but the trainer complaints
        smiles = pt.tensor(self.smiles_mm[idx], dtype=pt.int64).unsqueeze(0)
        Q = pt.tensor(self.Q_mm[idx], dtype=pt.float32).view(
            1, 1
        )
        tox_labels = pt.tensor(self.tox_labels_mm[idx], dtype=pt.float32).unsqueeze(0)
        return Data(z=z, az=atom_props, pos=pos, y=y, smiles=smiles, Q=Q, tox_labels=tox_labels)



sider_dict = [' familial and genetic disorders"',
 ' malignant and unspecified (incl cysts and polyps)"',
 ' puerperium and perinatal conditions"',
 ' thoracic and mediastinal disorders"',
 '"congenital',
 '"neoplasms benign',
 '"pregnancy',
 '"respiratory',
 'blood and lymphatic system disorders',
 'endocrine disorders',
 'eye disorders',
 'gastrointestinal disorders',
 'general disorders and administration site conditions',
 'hepatobiliary disorders',
 'immune system disorders',
 'infections and infestations',
 'investigations',
 'metabolism and nutrition disorders',
 'musculoskeletal and connective tissue disorders',
 'product issues',
 'psychiatric disorders',
 'renal and urinary disorders',
 'reproductive system and breast disorders',
 'skin and subcutaneous tissue disorders',
 'social circumstances',
 'surgical and medical procedures',
 'vascular disorders']

toxcast_dict = ['acea_t47d_80hr_negative',
 'acea_t47d_80hr_positive',
 'apr_hepat_apoptosis_24hr_up',
 'apr_hepat_apoptosis_48hr_up',
 'apr_hepat_cellloss_24hr_dn',
 'apr_hepat_cellloss_48hr_dn',
 'apr_hepat_dnadamage_24hr_up',
 'apr_hepat_dnadamage_48hr_up',
 'apr_hepat_dnatexture_24hr_up',
 'apr_hepat_dnatexture_48hr_up',
 'apr_hepat_mitofxni_1hr_dn',
 'apr_hepat_mitofxni_24hr_dn',
 'apr_hepat_mitofxni_48hr_dn',
 'apr_hepat_nuclearsize_24hr_dn',
 'apr_hepat_nuclearsize_48hr_dn',
 'apr_hepat_steatosis_24hr_up',
 'apr_hepat_steatosis_48hr_up',
 'apr_hepg2_cellcyclearrest_24h_dn',
 'apr_hepg2_cellcyclearrest_24h_up',
 'apr_hepg2_cellcyclearrest_72h_dn',
 'apr_hepg2_cellloss_24h_dn',
 'apr_hepg2_cellloss_72h_dn',
 'apr_hepg2_microtubulecsk_24h_dn',
 'apr_hepg2_microtubulecsk_24h_up',
 'apr_hepg2_microtubulecsk_72h_dn',
 'apr_hepg2_microtubulecsk_72h_up',
 'apr_hepg2_mitomass_24h_dn',
 'apr_hepg2_mitomass_24h_up',
 'apr_hepg2_mitomass_72h_dn',
 'apr_hepg2_mitomass_72h_up',
 'apr_hepg2_mitomembpot_1h_dn',
 'apr_hepg2_mitomembpot_24h_dn',
 'apr_hepg2_mitomembpot_72h_dn',
 'apr_hepg2_mitoticarrest_24h_up',
 'apr_hepg2_mitoticarrest_72h_up',
 'apr_hepg2_nuclearsize_24h_dn',
 'apr_hepg2_nuclearsize_72h_dn',
 'apr_hepg2_nuclearsize_72h_up',
 'apr_hepg2_oxidativestress_24h_up',
 'apr_hepg2_oxidativestress_72h_up',
 'apr_hepg2_p53act_24h_up',
 'apr_hepg2_p53act_72h_up',
 'apr_hepg2_stresskinase_1h_up',
 'apr_hepg2_stresskinase_24h_up',
 'apr_hepg2_stresskinase_72h_up',
 'atg_ahr_cis_dn',
 'atg_ahr_cis_up',
 'atg_ap_1_cis_dn',
 'atg_ap_1_cis_up',
 'atg_ap_2_cis_dn',
 'atg_ap_2_cis_up',
 'atg_ar_trans_dn',
 'atg_ar_trans_up',
 'atg_bre_cis_dn',
 'atg_bre_cis_up',
 'atg_c_ebp_cis_dn',
 'atg_c_ebp_cis_up',
 'atg_car_trans_dn',
 'atg_car_trans_up',
 'atg_cmv_cis_dn',
 'atg_cmv_cis_up',
 'atg_cre_cis_dn',
 'atg_cre_cis_up',
 'atg_dr4_lxr_cis_dn',
 'atg_dr4_lxr_cis_up',
 'atg_dr5_cis_dn',
 'atg_dr5_cis_up',
 'atg_e2f_cis_dn',
 'atg_e2f_cis_up',
 'atg_e_box_cis_dn',
 'atg_e_box_cis_up',
 'atg_egr_cis_up',
 'atg_era_trans_up',
 'atg_ere_cis_dn',
 'atg_ere_cis_up',
 'atg_erra_trans_dn',
 'atg_errg_trans_dn',
 'atg_errg_trans_up',
 'atg_ets_cis_dn',
 'atg_ets_cis_up',
 'atg_foxa2_cis_dn',
 'atg_foxa2_cis_up',
 'atg_foxo_cis_dn',
 'atg_foxo_cis_up',
 'atg_fxr_trans_up',
 'atg_gal4_trans_dn',
 'atg_gata_cis_dn',
 'atg_gata_cis_up',
 'atg_gli_cis_dn',
 'atg_gli_cis_up',
 'atg_gr_trans_dn',
 'atg_gr_trans_up',
 'atg_gre_cis_dn',
 'atg_gre_cis_up',
 'atg_hif1a_cis_dn',
 'atg_hif1a_cis_up',
 'atg_hnf4a_trans_dn',
 'atg_hnf4a_trans_up',
 'atg_hnf6_cis_dn',
 'atg_hnf6_cis_up',
 'atg_hse_cis_dn',
 'atg_hse_cis_up',
 'atg_ir1_cis_dn',
 'atg_ir1_cis_up',
 'atg_isre_cis_dn',
 'atg_isre_cis_up',
 'atg_lxra_trans_dn',
 'atg_lxra_trans_up',
 'atg_lxrb_trans_dn',
 'atg_lxrb_trans_up',
 'atg_m_06_trans_up',
 'atg_m_19_cis_dn',
 'atg_m_19_trans_dn',
 'atg_m_19_trans_up',
 'atg_m_32_cis_dn',
 'atg_m_32_cis_up',
 'atg_m_32_trans_dn',
 'atg_m_32_trans_up',
 'atg_m_61_trans_up',
 'atg_mre_cis_up',
 'atg_myb_cis_dn',
 'atg_myb_cis_up',
 'atg_myc_cis_dn',
 'atg_myc_cis_up',
 'atg_nf_kb_cis_dn',
 'atg_nf_kb_cis_up',
 'atg_nfi_cis_dn',
 'atg_nfi_cis_up',
 'atg_nrf1_cis_dn',
 'atg_nrf1_cis_up',
 'atg_nrf2_are_cis_dn',
 'atg_nrf2_are_cis_up',
 'atg_nurr1_trans_dn',
 'atg_nurr1_trans_up',
 'atg_oct_mlp_cis_dn',
 'atg_oct_mlp_cis_up',
 'atg_p53_cis_dn',
 'atg_p53_cis_up',
 'atg_pax6_cis_up',
 'atg_pbrem_cis_dn',
 'atg_pbrem_cis_up',
 'atg_ppara_trans_dn',
 'atg_ppara_trans_up',
 'atg_ppard_trans_up',
 'atg_pparg_trans_up',
 'atg_ppre_cis_dn',
 'atg_ppre_cis_up',
 'atg_pxr_trans_dn',
 'atg_pxr_trans_up',
 'atg_pxre_cis_dn',
 'atg_pxre_cis_up',
 'atg_rara_trans_dn',
 'atg_rara_trans_up',
 'atg_rarb_trans_dn',
 'atg_rarb_trans_up',
 'atg_rarg_trans_dn',
 'atg_rarg_trans_up',
 'atg_rorb_trans_dn',
 'atg_rore_cis_dn',
 'atg_rore_cis_up',
 'atg_rorg_trans_dn',
 'atg_rorg_trans_up',
 'atg_rxra_trans_dn',
 'atg_rxra_trans_up',
 'atg_rxrb_trans_dn',
 'atg_rxrb_trans_up',
 'atg_sox_cis_dn',
 'atg_sox_cis_up',
 'atg_sp1_cis_dn',
 'atg_sp1_cis_up',
 'atg_srebp_cis_dn',
 'atg_srebp_cis_up',
 'atg_stat3_cis_dn',
 'atg_stat3_cis_up',
 'atg_ta_cis_dn',
 'atg_ta_cis_up',
 'atg_tal_cis_dn',
 'atg_tal_cis_up',
 'atg_tcf_b_cat_cis_dn',
 'atg_tcf_b_cat_cis_up',
 'atg_tgfb_cis_dn',
 'atg_tgfb_cis_up',
 'atg_thra1_trans_dn',
 'atg_thra1_trans_up',
 'atg_vdr_trans_dn',
 'atg_vdr_trans_up',
 'atg_vdre_cis_dn',
 'atg_vdre_cis_up',
 'atg_xbp1_cis_dn',
 'atg_xbp1_cis_up',
 'atg_xtt_cytotoxicity_up',
 'bsk_3c_eselectin_down',
 'bsk_3c_hladr_down',
 'bsk_3c_icam1_down',
 'bsk_3c_il8_down',
 'bsk_3c_mcp1_down',
 'bsk_3c_mig_down',
 'bsk_3c_proliferation_down',
 'bsk_3c_srb_down',
 'bsk_3c_thrombomodulin_down',
 'bsk_3c_thrombomodulin_up',
 'bsk_3c_tissuefactor_down',
 'bsk_3c_tissuefactor_up',
 'bsk_3c_upar_down',
 'bsk_3c_vcam1_down',
 'bsk_3c_vis_down',
 'bsk_4h_eotaxin3_down',
 'bsk_4h_mcp1_down',
 'bsk_4h_pselectin_down',
 'bsk_4h_pselectin_up',
 'bsk_4h_srb_down',
 'bsk_4h_upar_down',
 'bsk_4h_upar_up',
 'bsk_4h_vcam1_down',
 'bsk_4h_vegfrii_down',
 'bsk_be3c_hladr_down',
 'bsk_be3c_il1a_down',
 'bsk_be3c_ip10_down',
 'bsk_be3c_mig_down',
 'bsk_be3c_mmp1_down',
 'bsk_be3c_mmp1_up',
 'bsk_be3c_pai1_down',
 'bsk_be3c_srb_down',
 'bsk_be3c_tgfb1_down',
 'bsk_be3c_tpa_down',
 'bsk_be3c_upa_down',
 'bsk_be3c_upar_down',
 'bsk_be3c_upar_up',
 'bsk_casm3c_hladr_down',
 'bsk_casm3c_il6_down',
 'bsk_casm3c_il6_up',
 'bsk_casm3c_il8_down',
 'bsk_casm3c_ldlr_down',
 'bsk_casm3c_ldlr_up',
 'bsk_casm3c_mcp1_down',
 'bsk_casm3c_mcp1_up',
 'bsk_casm3c_mcsf_down',
 'bsk_casm3c_mcsf_up',
 'bsk_casm3c_mig_down',
 'bsk_casm3c_proliferation_down',
 'bsk_casm3c_proliferation_up',
 'bsk_casm3c_saa_down',
 'bsk_casm3c_saa_up',
 'bsk_casm3c_srb_down',
 'bsk_casm3c_thrombomodulin_down',
 'bsk_casm3c_thrombomodulin_up',
 'bsk_casm3c_tissuefactor_down',
 'bsk_casm3c_upar_down',
 'bsk_casm3c_upar_up',
 'bsk_casm3c_vcam1_down',
 'bsk_casm3c_vcam1_up',
 'bsk_hdfcgf_collageniii_down',
 'bsk_hdfcgf_egfr_down',
 'bsk_hdfcgf_egfr_up',
 'bsk_hdfcgf_il8_down',
 'bsk_hdfcgf_ip10_down',
 'bsk_hdfcgf_mcsf_down',
 'bsk_hdfcgf_mig_down',
 'bsk_hdfcgf_mmp1_down',
 'bsk_hdfcgf_mmp1_up',
 'bsk_hdfcgf_pai1_down',
 'bsk_hdfcgf_proliferation_down',
 'bsk_hdfcgf_srb_down',
 'bsk_hdfcgf_timp1_down',
 'bsk_hdfcgf_vcam1_down',
 'bsk_kf3ct_icam1_down',
 'bsk_kf3ct_il1a_down',
 'bsk_kf3ct_ip10_down',
 'bsk_kf3ct_ip10_up',
 'bsk_kf3ct_mcp1_down',
 'bsk_kf3ct_mcp1_up',
 'bsk_kf3ct_mmp9_down',
 'bsk_kf3ct_srb_down',
 'bsk_kf3ct_tgfb1_down',
 'bsk_kf3ct_timp2_down',
 'bsk_kf3ct_upa_down',
 'bsk_lps_cd40_down',
 'bsk_lps_eselectin_down',
 'bsk_lps_eselectin_up',
 'bsk_lps_il1a_down',
 'bsk_lps_il1a_up',
 'bsk_lps_il8_down',
 'bsk_lps_il8_up',
 'bsk_lps_mcp1_down',
 'bsk_lps_mcsf_down',
 'bsk_lps_pge2_down',
 'bsk_lps_pge2_up',
 'bsk_lps_srb_down',
 'bsk_lps_tissuefactor_down',
 'bsk_lps_tissuefactor_up',
 'bsk_lps_tnfa_down',
 'bsk_lps_tnfa_up',
 'bsk_lps_vcam1_down',
 'bsk_sag_cd38_down',
 'bsk_sag_cd40_down',
 'bsk_sag_cd69_down',
 'bsk_sag_eselectin_down',
 'bsk_sag_eselectin_up',
 'bsk_sag_il8_down',
 'bsk_sag_il8_up',
 'bsk_sag_mcp1_down',
 'bsk_sag_mig_down',
 'bsk_sag_pbmccytotoxicity_down',
 'bsk_sag_pbmccytotoxicity_up',
 'bsk_sag_proliferation_down',
 'bsk_sag_srb_down',
 'ceetox_h295r_11dcort_dn',
 'ceetox_h295r_andr_dn',
 'ceetox_h295r_cortisol_dn',
 'ceetox_h295r_doc_dn',
 'ceetox_h295r_doc_up',
 'ceetox_h295r_estradiol_dn',
 'ceetox_h295r_estradiol_up',
 'ceetox_h295r_estrone_dn',
 'ceetox_h295r_estrone_up',
 'ceetox_h295r_ohpreg_up',
 'ceetox_h295r_ohprog_dn',
 'ceetox_h295r_ohprog_up',
 'ceetox_h295r_prog_up',
 'ceetox_h295r_testo_dn',
 'cld_abcb1_48hr',
 'cld_abcg2_48hr',
 'cld_cyp1a1_24hr',
 'cld_cyp1a1_48hr',
 'cld_cyp1a1_6hr',
 'cld_cyp1a2_24hr',
 'cld_cyp1a2_48hr',
 'cld_cyp1a2_6hr',
 'cld_cyp2b6_24hr',
 'cld_cyp2b6_48hr',
 'cld_cyp2b6_6hr',
 'cld_cyp3a4_24hr',
 'cld_cyp3a4_48hr',
 'cld_cyp3a4_6hr',
 'cld_gsta2_48hr',
 'cld_sult2a_24hr',
 'cld_sult2a_48hr',
 'cld_ugt1a1_24hr',
 'cld_ugt1a1_48hr',
 'ncct_hek293t_celltiterglo',
 'ncct_quantilum_inhib_2_dn',
 'ncct_quantilum_inhib_dn',
 'ncct_tpo_aur_dn',
 'ncct_tpo_gua_dn',
 'nheerl_zf_144hpf_teratoscore_up',
 'nvs_adme_hcyp19a1',
 'nvs_adme_hcyp1a1',
 'nvs_adme_hcyp1a2',
 'nvs_adme_hcyp2a6',
 'nvs_adme_hcyp2b6',
 'nvs_adme_hcyp2c19',
 'nvs_adme_hcyp2c9',
 'nvs_adme_hcyp2d6',
 'nvs_adme_hcyp3a4',
 'nvs_adme_hcyp4f12',
 'nvs_adme_rcyp2c12',
 'nvs_enz_hache',
 'nvs_enz_hampka1',
 'nvs_enz_haura',
 'nvs_enz_hbace',
 'nvs_enz_hcasp5',
 'nvs_enz_hck1d',
 'nvs_enz_hdusp3',
 'nvs_enz_helastase',
 'nvs_enz_hes',
 'nvs_enz_hfgfr1',
 'nvs_enz_hgsk3b',
 'nvs_enz_hmmp1',
 'nvs_enz_hmmp13',
 'nvs_enz_hmmp2',
 'nvs_enz_hmmp3',
 'nvs_enz_hmmp7',
 'nvs_enz_hmmp9',
 'nvs_enz_hpde10',
 'nvs_enz_hpde4a1',
 'nvs_enz_hpde5',
 'nvs_enz_hpi3ka',
 'nvs_enz_hpten',
 'nvs_enz_hptpn11',
 'nvs_enz_hptpn12',
 'nvs_enz_hptpn13',
 'nvs_enz_hptpn9',
 'nvs_enz_hptprc',
 'nvs_enz_hsirt1',
 'nvs_enz_hsirt2',
 'nvs_enz_htrka',
 'nvs_enz_hvegfr2',
 'nvs_enz_ocox1',
 'nvs_enz_ocox2',
 'nvs_enz_rabi2c',
 'nvs_enz_rache',
 'nvs_enz_rcnos',
 'nvs_enz_rmaoac',
 'nvs_enz_rmaoap',
 'nvs_enz_rmaobc',
 'nvs_enz_rmaobp',
 'nvs_gpcr_bador_nonselective',
 'nvs_gpcr_bdr_nonselective',
 'nvs_gpcr_g5ht4',
 'nvs_gpcr_gh2',
 'nvs_gpcr_gltb4',
 'nvs_gpcr_gltd4',
 'nvs_gpcr_gmperipheral_nonselective',
 'nvs_gpcr_gopiatek',
 'nvs_gpcr_h5ht2a',
 'nvs_gpcr_h5ht5a',
 'nvs_gpcr_h5ht6',
 'nvs_gpcr_h5ht7',
 'nvs_gpcr_hadora1',
 'nvs_gpcr_hadora2a',
 'nvs_gpcr_hadra2a',
 'nvs_gpcr_hadra2c',
 'nvs_gpcr_hadrb1',
 'nvs_gpcr_hadrb2',
 'nvs_gpcr_hadrb3',
 'nvs_gpcr_hat1',
 'nvs_gpcr_hdrd1',
 'nvs_gpcr_hdrd2s',
 'nvs_gpcr_hdrd4.4',
 'nvs_gpcr_hh1',
 'nvs_gpcr_hltb4_blt1',
 'nvs_gpcr_hm1',
 'nvs_gpcr_hm2',
 'nvs_gpcr_hm3',
 'nvs_gpcr_hm4',
 'nvs_gpcr_hnk2',
 'nvs_gpcr_hopiate_d1',
 'nvs_gpcr_hopiate_mu',
 'nvs_gpcr_htxa2',
 'nvs_gpcr_p5ht2c',
 'nvs_gpcr_r5ht1_nonselective',
 'nvs_gpcr_r5ht_nonselective',
 'nvs_gpcr_rabpaf',
 'nvs_gpcr_radra1_nonselective',
 'nvs_gpcr_radra1b',
 'nvs_gpcr_radra2_nonselective',
 'nvs_gpcr_radrb_nonselective',
 'nvs_gpcr_rmadra2b',
 'nvs_gpcr_rnk1',
 'nvs_gpcr_rnk3',
 'nvs_gpcr_ropiate_nonselective',
 'nvs_gpcr_ropiate_nonselectivena',
 'nvs_gpcr_rsst',
 'nvs_gpcr_rtrh',
 'nvs_gpcr_rv1',
 'nvs_ic_hkhergch',
 'nvs_ic_rcabtzchl',
 'nvs_ic_rcadhprch_l',
 'nvs_ic_rnach_site2',
 'nvs_lgic_bgabara1',
 'nvs_lgic_h5ht3',
 'nvs_lgic_hnnr_nbungsens',
 'nvs_lgic_rgabar_nonselective',
 'nvs_lgic_rnnr_bungsens',
 'nvs_mp_hpbr',
 'nvs_mp_rpbr',
 'nvs_nr_ber',
 'nvs_nr_bpr',
 'nvs_nr_car',
 'nvs_nr_har',
 'nvs_nr_hcar_antagonist',
 'nvs_nr_her',
 'nvs_nr_hfxr_agonist',
 'nvs_nr_hfxr_antagonist',
 'nvs_nr_hgr',
 'nvs_nr_hppara',
 'nvs_nr_hpparg',
 'nvs_nr_hpr',
 'nvs_nr_hpxr',
 'nvs_nr_hrar_antagonist',
 'nvs_nr_hrara_agonist',
 'nvs_nr_htra_antagonist',
 'nvs_nr_mera',
 'nvs_nr_rar',
 'nvs_nr_rmr',
 'nvs_or_gsigma_nonselective',
 'nvs_tr_gdat',
 'nvs_tr_hadot',
 'nvs_tr_hdat',
 'nvs_tr_hnet',
 'nvs_tr_hsert',
 'nvs_tr_rnet',
 'nvs_tr_rsert',
 'nvs_tr_rvmat2',
 'ot_ar_areluc_ag_1440',
 'ot_ar_arsrc1_0480',
 'ot_ar_arsrc1_0960',
 'ot_er_eraera_0480',
 'ot_er_eraera_1440',
 'ot_er_eraerb_0480',
 'ot_er_eraerb_1440',
 'ot_er_erberb_0480',
 'ot_er_erberb_1440',
 'ot_era_eregfp_0120',
 'ot_era_eregfp_0480',
 'ot_fxr_fxrsrc1_0480',
 'ot_fxr_fxrsrc1_1440',
 'ot_nurr1_nurr1rxra_0480',
 'ot_nurr1_nurr1rxra_1440',
 'tanguay_zf_120hpf_activityscore',
 'tanguay_zf_120hpf_axis_up',
 'tanguay_zf_120hpf_brai_up',
 'tanguay_zf_120hpf_cfin_up',
 'tanguay_zf_120hpf_circ_up',
 'tanguay_zf_120hpf_eye_up',
 'tanguay_zf_120hpf_jaw_up',
 'tanguay_zf_120hpf_mort_up',
 'tanguay_zf_120hpf_otic_up',
 'tanguay_zf_120hpf_pe_up',
 'tanguay_zf_120hpf_pfin_up',
 'tanguay_zf_120hpf_pig_up',
 'tanguay_zf_120hpf_snou_up',
 'tanguay_zf_120hpf_somi_up',
 'tanguay_zf_120hpf_swim_up',
 'tanguay_zf_120hpf_tr_up',
 'tanguay_zf_120hpf_trun_up',
 'tanguay_zf_120hpf_yse_up',
 'tox21_ahr_luc_agonist',
 'tox21_ar_bla_agonist_ch1',
 'tox21_ar_bla_agonist_ch2',
 'tox21_ar_bla_agonist_ratio',
 'tox21_ar_bla_antagonist_ch1',
 'tox21_ar_bla_antagonist_ch2',
 'tox21_ar_bla_antagonist_ratio',
 'tox21_ar_bla_antagonist_viability',
 'tox21_ar_luc_mdakb2_agonist',
 'tox21_ar_luc_mdakb2_antagonist',
 'tox21_ar_luc_mdakb2_antagonist2',
 'tox21_are_bla_agonist_ch1',
 'tox21_are_bla_agonist_ch2',
 'tox21_are_bla_agonist_ratio',
 'tox21_are_bla_agonist_viability',
 'tox21_aromatase_inhibition',
 'tox21_autofluor_hek293_cell_blue',
 'tox21_autofluor_hek293_media_blue',
 'tox21_autofluor_hepg2_cell_blue',
 'tox21_autofluor_hepg2_cell_green',
 'tox21_autofluor_hepg2_media_blue',
 'tox21_autofluor_hepg2_media_green',
 'tox21_elg1_luc_agonist',
 'tox21_era_bla_agonist_ch1',
 'tox21_era_bla_agonist_ch2',
 'tox21_era_bla_agonist_ratio',
 'tox21_era_bla_antagonist_ch1',
 'tox21_era_bla_antagonist_ch2',
 'tox21_era_bla_antagonist_ratio',
 'tox21_era_bla_antagonist_viability',
 'tox21_era_luc_bg1_agonist',
 'tox21_era_luc_bg1_antagonist',
 'tox21_esre_bla_ch1',
 'tox21_esre_bla_ch2',
 'tox21_esre_bla_ratio',
 'tox21_esre_bla_viability',
 'tox21_fxr_bla_agonist_ch2',
 'tox21_fxr_bla_agonist_ratio',
 'tox21_fxr_bla_antagonist_ch1',
 'tox21_fxr_bla_antagonist_ch2',
 'tox21_fxr_bla_antagonist_ratio',
 'tox21_fxr_bla_antagonist_viability',
 'tox21_gr_bla_agonist_ch1',
 'tox21_gr_bla_agonist_ch2',
 'tox21_gr_bla_agonist_ratio',
 'tox21_gr_bla_antagonist_ch2',
 'tox21_gr_bla_antagonist_ratio',
 'tox21_gr_bla_antagonist_viability',
 'tox21_hse_bla_agonist_ch1',
 'tox21_hse_bla_agonist_ch2',
 'tox21_hse_bla_agonist_ratio',
 'tox21_hse_bla_agonist_viability',
 'tox21_mmp_ratio_down',
 'tox21_mmp_ratio_up',
 'tox21_mmp_viability',
 'tox21_nfkb_bla_agonist_ch1',
 'tox21_nfkb_bla_agonist_ch2',
 'tox21_nfkb_bla_agonist_ratio',
 'tox21_nfkb_bla_agonist_viability',
 'tox21_p53_bla_p1_ch1',
 'tox21_p53_bla_p1_ch2',
 'tox21_p53_bla_p1_ratio',
 'tox21_p53_bla_p1_viability',
 'tox21_p53_bla_p2_ch1',
 'tox21_p53_bla_p2_ch2',
 'tox21_p53_bla_p2_ratio',
 'tox21_p53_bla_p2_viability',
 'tox21_p53_bla_p3_ch1',
 'tox21_p53_bla_p3_ch2',
 'tox21_p53_bla_p3_ratio',
 'tox21_p53_bla_p3_viability',
 'tox21_p53_bla_p4_ch1',
 'tox21_p53_bla_p4_ch2',
 'tox21_p53_bla_p4_ratio',
 'tox21_p53_bla_p4_viability',
 'tox21_p53_bla_p5_ch1',
 'tox21_p53_bla_p5_ch2',
 'tox21_p53_bla_p5_ratio',
 'tox21_p53_bla_p5_viability',
 'tox21_ppard_bla_agonist_ch1',
 'tox21_ppard_bla_agonist_ch2',
 'tox21_ppard_bla_agonist_ratio',
 'tox21_ppard_bla_agonist_viability',
 'tox21_ppard_bla_antagonist_ch1',
 'tox21_ppard_bla_antagonist_ratio',
 'tox21_ppard_bla_antagonist_viability',
 'tox21_pparg_bla_agonist_ch1',
 'tox21_pparg_bla_agonist_ch2',
 'tox21_pparg_bla_agonist_ratio',
 'tox21_pparg_bla_antagonist_ch1',
 'tox21_pparg_bla_antagonist_ratio',
 'tox21_pparg_bla_antagonist_viability',
 'tox21_tr_luc_gh3_agonist',
 'tox21_tr_luc_gh3_antagonist',
 'tox21_vdr_bla_agonist_ch2',
 'tox21_vdr_bla_agonist_ratio',
 'tox21_vdr_bla_agonist_viability',
 'tox21_vdr_bla_antagonist_ch1',
 'tox21_vdr_bla_antagonist_ratio',
 'tox21_vdr_bla_antagonist_viability']