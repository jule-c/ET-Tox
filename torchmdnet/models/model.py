import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import output_modules
from torchmdnet.models.utils import AccumulatedNormalization
import warnings
import logging 
from torchmdnet.models.smiles_transformer import TransformerEncoder
import numpy as np
import torch.nn.functional as F
from torchmdnet.models.energy_correction import *
from torchmdnet.models.utils import Distance


def create_model(args, prior_model=None, mean=None, std=None):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        num_atom_types=args["num_atom_types"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            use_total_charge=args["use_total_charge"],
            use_energy_feature=args["use_energy_feature"],
            use_smiles=args["use_smiles"],
            use_atom_props=args["use_atom_props"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # create output network
    linear_probing = 'LinearProbing' if args["linear_probing"] else ''
    energy_tox_multi_task = 'Scalar' if args["energy_tox_multi_task"] else ''
    smiles_only = args["use_smiles_only"]
    output_prefix = "Equivariant" if is_equivariant and not smiles_only else ""
    output_model = getattr(output_modules, output_prefix + linear_probing + energy_tox_multi_task + args["output_model"])(
        args["embedding_dimension"], args["activation"], output_channels=args["output_channels_toxicity"],
    )
    
    # create the charges output network
    output_model_charges = None
    if args['output_model_charges'] is not None:
        output_model_charges = getattr(output_modules, args["output_model_charges"])(
            args["embedding_dimension"], args["activation"],
        )
        
    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        reduce_op=args["reduce_op"],
        derivative=args["derivative"],
        max_z=args["max_z"],
        hidden_channels=args["embedding_dimension"],
        output_model_charges=output_model_charges,
        cutoff_upper=args["cutoff_upper"],
        long_range_cutoff=args["long_range_cutoff"],
        use_zbl_repulsion=args["use_zbl_repulsion"],
        use_electrostatics=args["use_electrostatics"],
        use_d4_dispersion=args["use_d4_dispersion"],
        compute_d4_atomic=args["use_d4_dispersion"],
        use_smiles=args["use_smiles"],
        context_length=args["max_len_smiles"],
        use_smiles_only=args["use_smiles_only"],
    )
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    args["test_only"] = False
    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value
    
    model = create_model(args)
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.to(device)

def load_representation_model(filepath, model, device="cpu"):
    ckpt = torch.load(filepath, map_location="cpu")
    state_dict = ckpt["state_dict"].copy()
    for k, v in ckpt["state_dict"].items():
        if not "representation_model" in k:
            del state_dict[k]
        else:
            if "pos_normalizer" in k:
                del state_dict[k]
            if "_embedding" in k and "neighbor_embedding" not in k:
                del state_dict[k]
    state_dict = {re.sub(r"^model.representation_model\.", "", k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device)

class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        reduce_op="add",
        derivative=False,
        hidden_channels=128,
        max_z=100,
        output_model_charges=None,
        cutoff_upper=None,
        long_range_cutoff=None,
        use_zbl_repulsion=False,
        use_electrostatics=False,
        use_d4_dispersion=False,
        compute_d4_atomic=False,
        use_smiles=False,
        context_length=175,
        use_smiles_only=False,
    ):
        super(TorchMD_Net, self).__init__()
        
        self.cutoff_upper = cutoff_upper
        self.long_range_cutoff = long_range_cutoff
        self.use_zbl_repulsion = use_zbl_repulsion
        self.use_electrostatics = use_electrostatics
        self.use_d4_dispersion = use_d4_dispersion
        self.compute_d4_atomic = compute_d4_atomic
        self.use_smiles = use_smiles
        self.use_smiles_only = use_smiles_only
        
        self.reduce_op = reduce_op
        self.derivative = derivative

        self.output_model = output_model
        
        if not use_smiles_only:
            self.representation_model = representation_model

            if self.use_smiles:
                self.smiles_encoder = TransformerEncoder(
                    context_length=context_length,
                    hidden_size=hidden_channels,
                    output_size=hidden_channels,
                )
            if self.use_zbl_repulsion:
                self.zbl_repulsion_energy = ZBLRepulsionEnergy()

            if self.use_electrostatics:
                self.electrostatic_energy = ElectrostaticEnergy(
                    cuton=0.25 * self.cutoff_upper,
                    cutoff=0.75 * self.cutoff_upper,
                    lr_cutoff=self.long_range_cutoff,
                )
            if self.use_d4_dispersion:
                self.d4_dispersion_energy = D4DispersionEnergy(cutoff=self.long_range_cutoff)

            if self.use_electrostatics or self.use_d4_dispersion:
                self.charges_mlp = output_model_charges
                self.distance = Distance(
                cutoff_lower=0.,
                cutoff_upper=self.long_range_cutoff,
                max_num_neighbors=300,
                return_vecs=True,
                loop=False,
            )

            self.reset_parameters()
        else:
            self.smiles_encoder = TransformerEncoder(
                    context_length=context_length,
                    hidden_size=hidden_channels,
                    output_size=hidden_channels,
                )

    def reset_parameters(self):
        
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        smiles: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        #assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch
        tox = None
        v = None
        
        if self.derivative:
            pos.requires_grad_(True)
        
        if not self.use_smiles_only:
        
            out_smiles = None
            if self.use_smiles:
                out_smiles = self.smiles_encoder(smiles)
            # run the potentially wrapped representation model
            x, v, z, pos, batch, edge_index, edge_weight = self.representation_model(z, pos, batch, y=y, Q=Q, smiles=out_smiles)

            # apply energy corrections
            ea_rep, ea_ele, ea_vdw = 0, 0, 0
            N = x.shape[0]
            num_batch = batch.unique()
            if self.use_zbl_repulsion:
                mask = edge_weight != 0.0
                cutoff_values = cutoff_function(edge_weight[mask], self.cutoff_upper)
                ea_rep = self.zbl_repulsion_energy(
                    N,
                    z.float(),
                    edge_weight[mask],
                    cutoff_values,
                    edge_index[1][mask],
                    edge_index[0][mask],
                )
                ea_rep = ea_rep.unsqueeze(1)

            # compute electrostatic contributions
            if self.use_electrostatics or self.use_d4_dispersion:
                charges = self.charges_mlp(x, v, z, pos, batch).squeeze()
                indices, r_ij, edge_vec = self.distance(pos, batch)
                idx_i, idx_j = indices[1], indices[0]

            if self.use_electrostatics:
                cell = None
                ea_ele = self.electrostatic_energy(
                    N, charges, r_ij, idx_i, idx_j, pos, cell, num_batch, batch
                )
                ea_ele = ea_ele.unsqueeze(1)

            # compute dispersion contributions
            if self.use_d4_dispersion:
                ea_vdw, pa, c6 = self.d4_dispersion_energy(
                    N, z, charges, r_ij, idx_i, idx_j, self.compute_d4_atomic
                )
                ea_vdw = ea_vdw.unsqueeze(1)

            x = x + ea_rep + ea_ele + ea_vdw
            
            # apply the output network
            if self.output_model.__class__.__name__ == "EquivariantScalarToxicity":
                x = self.output_model.pre_reduce(x, v, z, pos, batch)
                x = scatter(tox, batch, dim=0, reduce=self.reduce_op)
            else:
                x = self.output_model.pre_reduce(x, v, z, pos, batch)

            # aggregate atoms
            out = scatter(x, batch, dim=0, reduce=self.reduce_op)

            # apply output model after reduction
            out = self.output_model.post_reduce(out)

            return out, tox
        
        else:
            x = self.smiles_encoder(smiles)
            out = self.output_model.pre_reduce(x, v=None, z=z, pos=pos, batch=batch)
            return out, None
    
