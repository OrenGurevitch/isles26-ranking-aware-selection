# VERBATIM COPY — do not edit. Source: github.com/ezequieldlrosa/isles26 utils/eval_utils.py
# Pinned at commit 91965d0c5ae3a6a79ea9c52abd2756e67fd499b4, fetched 2026-07-27. This is the organizers' ranking code, kept here
# ONLY so frozen_isles/metrics/test.py can prove our wrapper returns identical numbers.
# To update: re-clone the repo, replace this file wholesale, and re-run the differential test.
#
# THIRD-PARTY CODE. Copyright (c) Ezequiel de la Rosa and the ISLES'26 organizers.
# Licensed under the MIT License [checked against the source repository 2026-08-07]. MIT permits
# redistribution provided this notice travels with the copy, which is why the notice is here and not
# only in a commit message. The rest of this repository is Apache 2.0; MIT is compatible with that,
# and this file remains under MIT rather than being relicensed.
__author__ = "Ezequiel de la Rosa"

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import warnings
from panoptica import (
    InputType,
    Panoptica_Evaluator,
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
)


def compute_pr_auc(ground_truth, prediction_map, empty_value=1.0):
    """
    Computes the voxel-wise PR-AUC (Precision-Recall AUC) score from a
    continuous prediction map (soft map, logits, or probability maps)
    against a binary ground-truth.

    Parameters
    ----------
    ground_truth : array-like, bool
        Binary ground-truth mask (any shape).
    prediction_map : array-like, float
        Continuous soft map of the same shape as `ground_truth`. Does not need
        to be bounded between 0 and 1.
    empty_value : scalar, float or np.nan, default=1.0
        The value returned if the ground_truth is entirely empty (all zeros).
        By default, it is set to 1.0 to reward correct empty predictions.

    Returns
    -------
    pr_auc_score : float
        The calculated Precision-Recall AUC score.
    """
    # 1. Ensure arrays are numpy arrays and flattened
    gt_flat = np.asarray(ground_truth).astype(np.bool_).ravel()
    pred_flat = np.asarray(prediction_map).astype(np.float32).ravel()

    if gt_flat.shape != pred_flat.shape:
        raise ValueError("Shape mismatch: ground_truth and prediction_map must match.")

    # 2. Check for the all-negative edge case (no lesion in GT)
    total_positives = np.sum(gt_flat)
    if total_positives == 0:
        # If the ground-truth is empty and the prediction map is completely uniform/flat,
        # the model correctly identified no lesion. Return the designated empty_value.
        if np.all(pred_flat == pred_flat[0]):
            return empty_value
        else:
            # If the ground truth is negative but predictions vary, Precision is 0.
            return 0.0

    # 3. Compute PR-AUC using sklearn
    try:
        precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
        pr_auc_score = auc(recall, precision)
    except ValueError:
        # Fallback if scikit-learn still raises an exception
        pr_auc_score = np.nan

    return pr_auc_score




def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = voxel_size.astype(float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_dice_f1_instance_difference(ground_truth, prediction, empty_value=1.0):
    """
    Computes the lesion-wise F1-score, instance count difference, and Dice score between two masks.

    Parameters
    ----------
    ground_truth : array-like, int
        Any array of arbitrary size. If not int, it will be converted.
    prediction: array-like, bool
        Any other array of identical size as 'ground_truth'. If not int, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    f1_score : float
        Instance-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty F1-Score = empty_value

    -------
    dice_score : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value
    -------
    instance_count_difference : int
        Absolute instance count difference as integer.
        Maximum similarity = 0
        No similarity = --> inf

    """

    ground_truth = np.asarray(ground_truth).astype(int)
    prediction = np.asarray(prediction).astype(int)

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(matching_threshold=0.25),
    )

    # FIXED LINE: Removed the `result, _` unpacking tuple
    result, _ = evaluator.evaluate(prediction, ground_truth, verbose=False)["ungrouped"]

    instance_count_difference = abs(
        result.num_ref_instances - result.num_pred_instances
    )

    if result.num_ref_instances == 0 and result.num_pred_instances == 0:
        f1_score = empty_value
        dice_score = empty_value
    else:
        f1_score = result.rq  # Recognition Quality / F1-score
        dice_score = result.global_bin_dsc

    return f1_score, instance_count_difference, dice_score
