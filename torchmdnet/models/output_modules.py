from abc import abstractmethod, ABCMeta
from typing import Optional
from torchmdnet.models.utils import act_class_mapping, GatedEquivariantBlock, GatedEquivariantLinearLayer
from torchmdnet.utils import atomic_masses
from torch_scatter import scatter
import torch
from torch import nn

__all__ = ["Scalar", "DipoleMoment", "ElectronicSpatialExtent"]


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, **kwargs):
        super(Scalar, self).__init__(allow_prior_model=allow_prior_model)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, **kwargs):
        super(EquivariantScalar, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0


class DipoleMoment(Scalar):
    def __init__(self, hidden_channels, activation="silu", **kwargs):
        super(DipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        atomic_mass = torch.from_numpy(atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class EquivariantDipoleMoment(EquivariantScalar):
    def __init__(self, hidden_channels, activation="silu", **kwargs):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        atomic_mass = torch.from_numpy(atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x + v.squeeze()

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class ElectronicSpatialExtent(OutputModel):
    def __init__(self, hidden_channels, activation="silu", **kwargs):
        super(ElectronicSpatialExtent, self).__init__(allow_prior_model=False)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )
        atomic_mass = torch.from_numpy(atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)

        x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
        return x


class EquivariantElectronicSpatialExtent(ElectronicSpatialExtent):
    pass


class EquivariantVector(EquivariantScalar):
    def __init__(self, hidden_channels, activation="silu", **kwargs):
        super(EquivariantVector, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()
    
class EquivariantVectorCategorical(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, max_z=5, **kwargs):
        super(EquivariantVectorCategorical, self).__init__(
            allow_prior_model=allow_prior_model
        )
        self.mixing = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, hidden_channels, activation=activation),
            ]
        )
        
        self.type_predictor = nn.Sequential(
                                    nn.Linear(hidden_channels, hidden_channels // 2),
                                    nn.SiLU(),
                                    nn.Linear(hidden_channels // 2, max_z))
        self.coord_predictor = nn.Linear(hidden_channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixing:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.mixing:
            x, v = layer(x, v)
        x = self.type_predictor(x)
        v = self.coord_predictor(v).squeeze()
        return x, v
    
class EquivariantScalarVector(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, **kwargs):
        super(EquivariantScalarVector, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x, v.squeeze()
    

class Toxicity(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=12):
        super(Toxicity, self).__init__(allow_prior_model=allow_prior_model)
        
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            act_class(),
        )
        
        self.toxicity_prediction = nn.Linear(hidden_channels, output_channels)

    def pre_reduce(self, x, v, z, pos, batch):
        x = self.output_network(x)
        tox_pred = self.toxicity_prediction(x)
        return tox_pred
    
class EquivariantToxicity(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=12):
        
        super(EquivariantToxicity, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels, hidden_channels, activation=activation),
            ]
        )
        self.down_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
        )
        self.toxicity_prediction = nn.Linear(hidden_channels // 2, output_channels)
        
            
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        x = x + 0*v.sum()
        x = self.down_projection(x)
        tox_pred = self.toxicity_prediction(x)
        return tox_pred
    
class EquivariantScalarToxicity(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=12):
        
        super(EquivariantScalarToxicity, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, hidden_channels // 2, activation=activation),
            ]
        )
        self.toxicity_prediction = nn.Linear(hidden_channels // 2, output_channels)
        #self.energy_prediction = nn.Linear(hidden_channels // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        x = x + 0*v.sum()
        tox_pred = self.toxicity_prediction(x)
        energy_pred = self.energy_prediction(x)
        return energy_pred, tox_pred
    
class ScalarToxicity(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=12):
        
        super(ScalarToxicity, self).__init__(allow_prior_model=allow_prior_model)

        self.toxicity_prediction = nn.Linear(hidden_channels, output_channels)
        #self.energy_prediction = nn.Linear(hidden_channels // 2, 1)


    def pre_reduce(self, x, v, z, pos, batch):

        tox_pred = self.toxicity_prediction(x)
        #energy_pred = self.energy_prediction(x)
        return tox_pred


class LinearProbingToxicity(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=12):
        
        super(LinearProbingToxicity, self).__init__(allow_prior_model=allow_prior_model)
        self.toxicity_prediction = nn.Linear(hidden_channels, output_channels)
        
    def pre_reduce(self, x, v, z, pos, batch):        
        tox_pred = self.toxicity_prediction(x)
        return tox_pred
    
class EquivariantLinearProbingToxicity(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=12):
        super(EquivariantLinearProbingToxicity, self).__init__(allow_prior_model=allow_prior_model)
        
        self.combine_vector_scalar = GatedEquivariantLinearLayer(hidden_channels, hidden_channels // 2)
        self.toxicity_prediction = nn.Linear(hidden_channels // 2, output_channels)

    def pre_reduce(self, x, v, z, pos, batch):
        x = self.combine_vector_scalar(x, v)
        tox_pred = self.toxicity_prediction(x)
        return tox_pred
    

class Charges(nn.Module):
    def __init__(self, hidden_channels, activation="silu"):
        super(Charges, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + 0*v.sum()
