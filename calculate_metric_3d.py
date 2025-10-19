import os
import numpy as np
from torch.utils.data import Dataset
import json
import argparse
import trimesh
import scipy as sp
import numpy as np
from tqdm import tqdm


class ImagePairDataset(Dataset):
    """Image pair dataset"""

    def __init__(
        self, 
        eval_results_dir,
        gt_dir,
        test_data_info_path,
        all_data_info_path,
        metric_data=None,
        wo_mixamo=False,
    ):

        with open(test_data_info_path, 'r') as f:
            test_data_info = json.load(f)

        with open(all_data_info_path, 'r') as f:
            all_data_info = json.load(f)

        self.image_pairs = []

        # add alpaca data
        for alpaca_key in test_data_info['alpaca']:
            if metric_data is not None and alpaca_key in metric_data['alpaca'] and 'C.D.' in metric_data['alpaca'][alpaca_key]:
                continue
            pred_glb_path = os.path.join(eval_results_dir, 'alpaca', alpaca_key, 'edit.glb')
            gt_glb_path = os.path.join(gt_dir, 'alpaca', alpaca_key, 'edit.glb')
            assert os.path.exists(pred_glb_path) and os.path.exists(gt_glb_path), pred_glb_path + gt_glb_path
            self.image_pairs.append([
                ('alpaca', alpaca_key),
                pred_glb_path,
                gt_glb_path,
            ])
                
        # add flux_edit data
        for flux_edit_key in test_data_info['flux_edit']:
            if metric_data is not None and flux_edit_key in metric_data['flux_edit'] and 'C.D.' in metric_data['flux_edit'][flux_edit_key]:
                continue
            pred_glb_path = os.path.join(eval_results_dir, 'flux_edit', flux_edit_key, 'edit.glb')
            gt_glb_path = os.path.join(gt_dir, 'flux_edit', flux_edit_key, 'edit.glb')
            assert os.path.exists(pred_glb_path) and os.path.exists(gt_glb_path), pred_glb_path + gt_glb_path
            self.image_pairs.append([
                ('flux_edit', flux_edit_key),
                pred_glb_path,
                gt_glb_path,
            ])

        # add mixamo data
        if not wo_mixamo:
            for mixamo_key in test_data_info['mixamo']:

                object_name, ori_idx, edit_idx = mixamo_key
                pred_key = f'{object_name}_{ori_idx}_{edit_idx}'

                if metric_data is not None and pred_key in metric_data['mixamo'] and 'C.D.' in metric_data['mixamo'][pred_key]:
                    continue

                gt_key = all_data_info['mixamo'][object_name][int(edit_idx)]['ss_latents_path'].split('/')[-1][:-4]
                pred_glb_path = os.path.join(eval_results_dir, 'mixamo', pred_key, 'edit.glb')
                gt_glb_path = os.path.join(gt_dir, 'mixamo', gt_key, 'ori.glb')
                assert os.path.exists(pred_glb_path) and os.path.exists(gt_glb_path), pred_glb_path + gt_glb_path
                self.image_pairs.append([
                    ('mixamo', pred_key),
                    pred_glb_path,
                    gt_glb_path,
                ])

        print(f'Total object num: {len(self.image_pairs)}')

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        return pair


def distance_field_helper(source, target, normals_src=None, normals_tgt=None):
    target_kdtree = sp.spatial.cKDTree(target)
    distances, idx = target_kdtree.query(source, workers=-1)

    if normals_src is not None and normals_tgt is not None:
        
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)

    else:
        normals_dot_product = np.array(
            [np.nan] * source.shape[0], dtype=np.float32)

    return distances, normals_dot_product


def compute_surface_metrics(mesh_pred, mesh_gt, eval_points=100000, tau_in_fscore=1e-2, eval_num=1):
    """Compute surface metrics (chamfer distance and f-score) for one example.
    Args:
    mesh: trimesh.Trimesh, the mesh to evaluate.
    Returns:
    chamfer: float, chamfer distance.
    fscore: float, f-score.
    """
    cd_l1_1e3_list = []
    normal_consistency_list = []
    fscore_list = []
    for _ in range(eval_num):
        # Chamfer
        point_gt, idx_gt = mesh_gt.sample(eval_points, return_index=True)
        normal_gt = mesh_gt.face_normals[idx_gt]
        point_gt = point_gt.astype(np.float32)

        point_pred, idx_pred = mesh_pred.sample(eval_points, return_index=True)
        normal_pred = mesh_pred.face_normals[idx_pred]
        point_pred = point_pred.astype(np.float32)

        dist_pred_to_gt, normal_pred_to_gt = distance_field_helper(point_pred, point_gt, normal_pred, normal_gt)
        dist_gt_to_pred, normal_gt_to_pred = distance_field_helper(point_gt, point_pred, normal_gt, normal_pred)

        chamfer_l1 = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)

        c_p2g = np.mean(dist_pred_to_gt)
        c_g2p = np.mean(dist_gt_to_pred)
        cd_l1_1e3 = ((c_p2g + c_g2p) / 2.0) * 1000.0

        normal_consistency = np.mean(normal_pred_to_gt) + np.mean(normal_gt_to_pred)

        # Fscore
        tau = tau_in_fscore
        eps = 1e-6

        #dist_pred_to_gt = (dist_pred_to_gt**2)
        #dist_gt_to_pred = (dist_gt_to_pred**2)

        prec_tau = (dist_pred_to_gt <= tau).astype(np.float32).mean() * 100.
        recall_tau = (dist_gt_to_pred <= tau).astype(np.float32).mean() * 100.

        fscore = (2 * prec_tau * recall_tau) / max(prec_tau + recall_tau, eps)

        cd_l1_1e3_list.append(cd_l1_1e3)
        normal_consistency_list.append(normal_consistency / 2.)
        fscore_list.append(fscore)

    # Following the tradition to scale chamfer distance up by 10.
    return np.mean(cd_l1_1e3_list), np.mean(normal_consistency_list), np.mean(fscore_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_results_dir', type=str, default='./work_dirs/eval_results/01')
    parser.add_argument('--dataset_root_dir', type=str, default='/path_to_3DEditVerse')
    parser.add_argument('--save_json_path', type=str, default=None)
    parser.add_argument('--wo_mixamo', action='store_true')
    args = parser.parse_args()

    args.save_json_path = os.path.join(args.eval_results_dir, 'eval_metric.json')

    if os.path.exists(args.save_json_path):
        with open(args.save_json_path, 'r') as file:
            metric_data = json.load(file)
    else:
        metric_data = {
            'alpaca': {
            },
            'flux_edit': {
            },
            'mixamo': {
            },
        }

    dataset = ImagePairDataset(
        eval_results_dir=args.eval_results_dir, 
        gt_dir=os.path.join(args.dataset_root_dir, 'test_data'),
        test_data_info_path=os.path.join(args.dataset_root_dir, 'test_data_info.json'),
        all_data_info_path=os.path.join(args.dataset_root_dir, 'dataset_info.json'),
        wo_mixamo=args.wo_mixamo, 
        metric_data=metric_data
    )

    for pair in tqdm(dataset):

        dataset_type, key = pair[0]
        pred_glb_path = pair[1]
        gt_glb_path = pair[2]

        mesh_pred = trimesh.load(pred_glb_path).to_mesh()
        mesh_gt = trimesh.load(gt_glb_path).to_mesh()

        cd, normal_consistency, fscore = compute_surface_metrics(mesh_pred, mesh_gt)
        if key not in metric_data[dataset_type]:
            metric_data[dataset_type][key] = {}
        metric_data[dataset_type][key]['C.D.'] = cd
        metric_data[dataset_type][key]['Normal Consistency'] = normal_consistency
        metric_data[dataset_type][key]['Fscore'] = fscore

    # save metric data to json
    with open(args.save_json_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(metric_data, indent=4))

    