import glob
import os

import numpy as np
import surface_distance

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def compute_dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return 1.0
        else:
            return 0.

    test, reference = confusion_matrix.test.astype(np.bool_), confusion_matrix.reference.astype(np.bool_)
    dice_coefficient = surface_distance.compute_dice_coefficient(test, reference)
    return dice_coefficient


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return np.nan
        else:
            return 0
    test, reference = confusion_matrix.test.astype(np.bool_), confusion_matrix.reference.astype(np.bool_)
    distance_obj = surface_distance.compute_surface_distances(test, reference, spacing_mm=voxel_spacing)
    average_surface_distance = surface_distance.compute_average_surface_distance(distance_obj)
    assd = (average_surface_distance[0] + average_surface_distance[1]) / 2.0
    return assd


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return np.nan
        else:
            return 0

    test, reference = confusion_matrix.test.astype(np.bool_), confusion_matrix.reference.astype(np.bool_)
    distance_obj = surface_distance.compute_surface_distances(test, reference, spacing_mm=voxel_spacing)
    hausdorff = surface_distance.compute_robust_hausdorff(distance_obj, 95)
    return hausdorff


def compute_seg_metrics(prediction, mask, spacing_mm, key=''):
    """Computes segmentation metrics.

    Args:
        prediction (ndarray): prediction array (binary)
        mask (ndarray): mask array (binary)
        spacing_mm (tuple&list): spacing, unit-mm
    Returns:
        metrics_dict: metrics dictionary
    """
    dice = compute_dice(mask, prediction)
    average_surface_distance = avg_surface_distance_symmetric(mask, prediction, voxel_spacing=spacing_mm)
    hausdorff = hausdorff_distance_95(mask, prediction, voxel_spacing=spacing_mm)
    recall_value = recall(prediction, mask)
    precision_value = precision(prediction, mask)

    metrics_dict = {
        f'{key}_Dice': round(dice, 3),
        f'{key}_ASD': round(average_surface_distance, 3),
        f'{key}_Haus95': round(hausdorff, 3),
        f'{key}_Recall': round(recall_value, 3),
        f'{key}_Precision': round(precision_value, 3),
    }
    return metrics_dict


def compute_metrics(mask_arr, pred_arr, spacing_mm, label_dict):
    """compute metricmetrics

    Args:
        mask_arr (ndArrray): mask array
        pred_arr (ndArrray): prediciton  array
        spacing_mm (tuple): spacing unit-mm
        label_dict (dict): label info what you want to compute. like {'stone': 1, 'kidney': 2}

    Returns:
        _type_: _description_
    """
    metrics_dict = {}

    for k in label_dict.keys():
        label_value = label_dict[k]
        tmp_pred_arr = pred_arr.copy()
        tmp_mask_arr = mask_arr.copy()

        tmp_pred_arr[tmp_pred_arr != label_value] = 0
        tmp_mask_arr[tmp_mask_arr != label_value] = 0

        tmp_pred_arr[tmp_pred_arr == label_value] = 1
        tmp_mask_arr[tmp_mask_arr == label_value] = 1

        tmp_metrics_dict = compute_seg_metrics(tmp_pred_arr, tmp_mask_arr, spacing_mm, key=k)
        metrics_dict.update(tmp_metrics_dict)

    return metrics_dict


def compute_nii_metrics(stage, masks, preds):
    '''通过预测结果和真实标注计算评价指标'''

    metrics_list = []
    for i in tqdm(range(len(masks))):
        mask_itk = sitk.ReadImage(masks[i])
        mask_arr = sitk.GetArrayFromImage(mask_itk)
        pred_itk = sitk.ReadImage(preds[i])
        pred_arr = sitk.GetArrayFromImage(pred_itk)

        itk_spacing = mask_itk.GetSpacing()
        spacing = [itk_spacing[2], itk_spacing[1], itk_spacing[0]]

        if stage == 'ps': # 平扫期指标计算，包括结石和肾轮廓
            label_dict = {'stone': 1, 'kidney': 2}
        elif stage == 'pz': # 皮质期指标计算，包括血管、髓质和皮质
            label_dict = {'artery': 1, 'medulla': 2, 'cortex': 3}
        elif stage == 'px': # 排泄期指标计算，包括集合系统和肾轮廓
            label_dict = {'collecting_system': 1, 'parenchyma': 2}
            mask_arr[mask_arr == 3] = 1
            pred_arr[pred_arr == 3] = 1
        elif stage == 'reg': # 配准指标计算，只计算肾轮廓
            label_dict = {'kidney': 1}
        elif stage == 'merge': # 合并图像指标计算，包括集合系统、肾轮廓、血管、髓质和肾结石，目前用不到
            label_dict = {'collecting_system': 1, 'kidney': 2, 'artery': 3, 'medulla': 4, 'stone': 5}
        elif stage == 'artery': # 只计算血管
            label_dict = {'artery': 1}

        else:
            raise Exception('stage error!')

        metrics_dict = {}
        # metrics_dict['case_name'] = masks[i].split('mask')[1][1:8]
        metrics_dict['case_name'] = masks[i].split('wd_')[-1][0:3]
        metrics_dict.update(compute_metrics(mask_arr, pred_arr, spacing, label_dict))
        metrics_list.append(metrics_dict)

    metrics_d = pd.DataFrame(metrics_list)

    return metrics_d


def compute_3stage_metrics(root_path):
    ''' 分别计算三个时期的指标 '''

    stages = ['ps', 'px', 'pz']
    for stage in stages:
        masks = sorted(glob.glob(os.path.join(root_path, "mask", "*", "*" + stage + "*mask1*")))
        preds = sorted(glob.glob(os.path.join(root_path, "pred", "*", "*" + stage + "*")))
        # json_path = os.path.join(root_path, stage + ".json")
        metrics_d = compute_nii_metrics(stage, masks, preds)
        # metrics_d.to_json(json_path)  # 生成json

        # 生成统计值
        excel_path = os.path.join(root_path, stage + ".xlsx")
        df = metrics_d.describe().loc[['mean', 'std']].T.apply(lambda x: round(x, 3))
        df['mean'] = df['mean'].astype(str)
        df['std'] = df['std'].astype(str)
        df['Mean ± Std'] = [' ± '.join(i) for i in df[['mean', 'std']].values]
        df.to_excel(excel_path)  # 生成excel

        # 生成排序值
        excel_path = os.path.join(root_path, stage + "_all.xlsx")
        metrics_d.to_excel(excel_path)  # 生成excel


def compute_manual_error(root_path):
    ''' 分别计算三个时期的人人误差 '''

    stages = ['ps', 'px', 'pz']
    for stage in stages:
        mask1 = sorted(glob.glob(os.path.join(root_path, "mask", "*", "*" + stage + "*mask1*")))
        mask2 = sorted(glob.glob(os.path.join(root_path, "mask", "*", "*" + stage + "*mask2*")))
        mask3 = sorted(glob.glob(os.path.join(root_path, "mask", "*", "*" + stage + "*mask3*")))
        metrics_d21 = compute_nii_metrics(stage, mask1, mask2)
        metrics_d31 = compute_nii_metrics(stage, mask1, mask3)
        metrics_d23 = compute_nii_metrics(stage, mask2, mask3)

        # 生成排序值
        excel_path = os.path.join(root_path, stage + "12all.xlsx")
        metrics_d21.to_excel(excel_path)
        excel_path = os.path.join(root_path, stage + "13all.xlsx")
        metrics_d31.to_excel(excel_path)
        excel_path = os.path.join(root_path, stage + "23all.xlsx")
        metrics_d23.to_excel(excel_path)

        # 生成统计值
        df21 = metrics_d21.describe().loc[['mean', 'std']].T.apply(lambda x: round(x, 3))
        df31 = metrics_d31.describe().loc[['mean', 'std']].T.apply(lambda x: round(x, 3))
        df = ((df21 + df31) / 2).apply(lambda x: round(x, 3))
        excel_path = os.path.join(root_path, stage + ".xlsx")
        df['mean'] = df['mean'].astype(str)
        df['std'] = df['std'].astype(str)
        df['Mean ± Std'] = [' ± '.join(i) for i in df[['mean', 'std']].values]
        df.to_excel(excel_path)  # 生成excel


def compute_single_metrics(root_path):
    ''' 计算单个标注的指标 '''

    stage = 'artery'
    json_path = os.path.join(root_path, "json", "rl6_new.json")
    df = pd.read_json(json_path)
    masks = []
    for _, row in df.iterrows():
        if row['dataset'] == 'test':
            masks.append(row['mask_path'])
    preds = sorted(glob.glob(os.path.join(root_path, "test_output", "*.nii.gz")))
    metrics_d = compute_nii_metrics(stage, masks, preds)

    # 生成统计值
    excel_path = os.path.join(root_path, "metrics", "nnunetv2_artery.xlsx")
    df = metrics_d.describe().loc[['mean', 'std']].T.apply(lambda x: round(x, 3))
    df['mean'] = df['mean'].astype(str)
    df['std'] = df['std'].astype(str)
    df['Mean ± Std'] = [' ± '.join(i) for i in df[['mean', 'std']].values]
    df.to_excel(excel_path)  # 生成excel

    # 生成排序值
    excel_path = os.path.join(root_path, "metrics", "nnunetv2_artery_all.xlsx")
    metrics_d.to_excel(excel_path)  # 生成excel


if __name__ == '__main__':

    # 要测试的项目目录
    root_path = '/workspace/data/rl'

    compute_single_metrics(root_path)
