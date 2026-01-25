from .rkd import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .proposed_loss import StrongerKD
from .contrastive import ContrastiveLoss
from .ours import ProposalLossWithDTW
from .ckd import CKD
from .emo import EMO
from .norm_kd import NormKD
from .span_propose_wo_hid_cross import SpanProposeCriterionWeightedWOHidCross
from .span_propose_wo_hid_intra import SpanProposeCriterionWeightedWOHidIntra
from .span_propose_wo_intra_cross import SpanProposeCriterionWeightedWOIntraCross
from .span_propose_wo_intra import SpanProposeCriterionWeightedWOIntra

criterion_list = {
    "rkd": ContrastiveLossWithRKD,
    "ours": ProposalLossWithDTW,
    "uld": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "contrastive": ContrastiveLoss,
    "dang_propose": StrongerKD,
    "ckd": CKD,
    "emo": EMO,
    "norm": NormKD,
    "span_propose_wo_hid_cross": SpanProposeCriterionWeightedWOHidCross,
    "span_propose_wo_hid_intra": SpanProposeCriterionWeightedWOHidIntra,
    "span_propose_wo_intra_cross": SpanProposeCriterionWeightedWOIntraCross,
    "span_propose_wo_intra": SpanProposeCriterionWeightedWOIntra
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)