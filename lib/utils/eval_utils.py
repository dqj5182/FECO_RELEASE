import numpy as np
from lib.core.config import cfg


def evaluation(outputs, targets_data, meta_info, mode='val', thres=0.5):
    eval_out = {}

    # GT
    foot_valid = meta_info['foot_valid'] is not None

    # Pred
    if cfg.DATASET.test_name not in ['OpenPose', 'InstaVariety']:
        contact_pred = outputs['contact_out'].sigmoid()[0].detach().cpu().numpy()
    else:
        contact_joint_openpose_out = outputs['contact_joint_openpose_out'].sigmoid()[0].detach().cpu().numpy() # Remove sigmoid when evaluating zero velocity

    # Error Calculate
    if foot_valid:
        # Contact Metrics
        if cfg.DATASET.test_name in ['OpenPose', 'InstaVariety']:
            cont_pre, cont_rec, cont_f1 = compute_contact_metrics(targets_data['contact_data']['contact_f_joint_openpose_2d'][0].detach().cpu().numpy(), contact_joint_openpose_out, foot_valid, thres=thres)
        else:
            cont_pre, cont_rec, cont_f1 = compute_contact_metrics(targets_data['contact_data']['contact_f'][0].detach().cpu().numpy(), contact_pred, foot_valid, thres=thres)
        eval_out['cont_pre'] = cont_pre
        eval_out['cont_rec'] = cont_rec
        eval_out['cont_f1'] = cont_f1

    return eval_out


def compute_contact_metrics(gt, pred, valid, thres=0.5):
    """
    Compute precision, recall, and f1 using NumPy
    """
    if valid:
        tp_num = np.sum(gt[pred >= thres])

        precision_denominator = np.sum(pred >= thres)
        recall_denominator = np.sum(gt)

        precision_ = tp_num / precision_denominator if precision_denominator > 0 else None
        recall_ = tp_num / recall_denominator if recall_denominator > 0 else None
        if precision_ is not None and recall_ is not None and (precision_ + recall_) > 0:
            f1_ = 2 * precision_ * recall_ / (precision_ + recall_)
        else:
            f1_ = None
    else:
        precision_ = None
        recall_ = None
        f1_ = None

    return precision_, recall_, f1_