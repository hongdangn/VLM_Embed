from .rkd import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .proposed_loss import StrongerKD
from .contrastive import ContrastiveLoss
from .ours import ProposalLossWithDTW
from .ckd import CKD

criterion_list = {
    "rkd": ContrastiveLossWithRKD,
    "ours": ProposalLossWithDTW,
    "uld": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "contrastive": ContrastiveLoss,
    "dang_propose": StrongerKD,
    "ckd": CKD,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)