import torch
from utils import diff_operators, quaternion

# uses real units


def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):#, switch_results1, switch_results2, switch_results3, switch_coef=1.0):
        #pde_mask = torch.logical_not(switch_mask)
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
            switch_loss=torch.abs(torch.Tensor([0])).sum()
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
                #diff_constraint_hom = diff_constraint_hom[pde_mask]
            #switch_loss=torch.abs(switch_results1-switch_results2).sum()*switch_coef
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepReach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
                return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                    "switch_loss": torch.abs(torch.Tensor([0])).sum()}
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                        "switch_loss": torch.abs(torch.Tensor([0])).sum()}
        elif dynamics.deepReach_model in ['exact_sin','exact_exp']:
            if torch.all(dirichlet_mask):
                # dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0 # pretrain the network to output zero
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                        "switch_loss": torch.abs(switch_results1-switch_results2).sum()*switch_coef}
            
        elif dynamics.deepReach_model == 'exact_diff':
            if torch.all(dirichlet_mask):
                dirichlet = output[0].squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                        "switch_loss": torch.abs(switch_results1-switch_results2).sum()*switch_coef}

        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                "switch_loss": torch.abs(torch.Tensor([0])).sum()}

    return brt_hjivi_loss

def init_brt_hjivi_aug_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_aug_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, -value + boundary_value)
        dirichlet = -value[dirichlet_mask] + boundary_value[dirichlet_mask]
        if dynamics.deepReach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
                return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                    'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                    "switch_loss": torch.abs(torch.Tensor([0])).sum()}
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                        "switch_loss": torch.abs(torch.Tensor([0])).sum()}
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                "switch_loss": torch.abs(torch.Tensor([0])).sum()}

    return brt_hjivi_aug_loss

def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask,output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepReach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    return brat_hjivi_loss
