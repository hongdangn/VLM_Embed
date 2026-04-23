from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .span_propose import SpanProposeCriterion
from .span_propose_attn import SpanProposeCriterionWeighted
from .span_propose_attn_only_phrase import SpanProposeCriterionWeightedOnlyPhrase
from .gvendi import GVendiVLMCriterion
from .gvendi_topology_extract_checker import GvendiTopologyExtract

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
    "em_kd_llava_ov": EMKDLLavaLoss,
    "span_propose": SpanProposeCriterion,
    "span_propose_attn": SpanProposeCriterionWeighted,
    "span_propose_attn_only_phrase": SpanProposeCriterionWeightedOnlyPhrase,
    "gvendi_phase2": GVendiVLMCriterion,
    "gvendi_phase1": GvendiTopologyExtract
}

def build_criterion(data_args, training_args, distiller):
    if training_args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {training_args.kd_loss_type} not found.")
    
    if "gvendi" in training_args.kd_loss_type:
        return criterion_list[training_args.kd_loss_type](data_args, training_args, distiller)
    return criterion_list[training_args.kd_loss_type](training_args)