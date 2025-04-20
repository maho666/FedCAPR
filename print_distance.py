import torch
import logging
import argparse
import numpy as np
import sys
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure
from sklearn.cluster import DBSCAN
# from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
from print_distance_utils.faiss_rerank import faiss_compute_jaccard_dist
import scipy.io as sio 
torch.autograd.set_detect_anomaly(True)
from reid.models import stb_net
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import UnsupervisedTargetPreprocessor
from dataset import prepare_train_data
logger = logging.getLogger(__name__)



def calculate_distance(network, propagate_loader, target, device, weight, k1=20, k2=6):
    network.eval()
    features = []
    global_labels = []
    all_pids = []

    with torch.no_grad():
        for c, data in enumerate(propagate_loader):
            images = data[0].to(device) # data = (img, fname, pid, img_idx, camid)
            pid = data[2]
            g_label = data[3]
            embed_feat = network(images)
            features.append(embed_feat.cpu())
            all_pids.append(pid)
            global_labels.append(g_label)
    features = torch.cat(features, dim=0).numpy()
    global_labels = torch.cat(global_labels, dim=0).numpy()
    all_pids = torch.cat(all_pids, dim=0).numpy()
    # if needed, average camera-style transferred image features
    new_features = []
    new_pids = []
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        new_features.append(np.mean(features[idx], axis=0))
        new_pids.append(all_pids[idx])
    new_features = np.array(new_features)
    new_pids = np.array(new_pids).reshape(-1)
    del features, all_pids

    # compute distance W
    new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # l2-normalize
    W = faiss_compute_jaccard_dist(torch.from_numpy(new_features), weight, k1=k1, k2=k2, print_flag=False)

    # modified W dist matrix for dataset append version
    # if weight == 1:
    #     expended_W = W
    # else:
    #     N = W.shape[0] * weight
    #     expended_W = np.zeros((N, N), dtype=np.float32)
    #     for i in range(W.shape[0]):
    #         base_dim0 = i * weight
    #         for j in range(W.shape[1]):
    #             base_dim1 = j * weight
    #             expended_W[base_dim0:(base_dim0+weight), base_dim1:(base_dim1+weight)] = W[i,j]

    # epsbytest
    # if target == "ilids" or target == "3dpes":
    #     cluster = DBSCAN(eps=0.35, min_samples=4, metric='precomputed', n_jobs=8)
    # elif target == "viper":
    #     cluster = DBSCAN(eps=0.4, min_samples=4, metric='precomputed', n_jobs=8)
    # else:
    #     cluster = DBSCAN(eps=0.5, min_samples=4, metric='precomputed', n_jobs=8)
    # epsbytest__v1
    # if target == "ilids" or target == "3dpes" or target == "viper":
    #     cluster = DBSCAN(eps=0.35, min_samples=4, metric='precomputed', n_jobs=8)
    # elif target == "cuhk01":
    #     cluster = DBSCAN(eps=0.35, min_samples=4, metric='precomputed', n_jobs=8)
    # else:
    # epsbytest__v2
    # if target == "ilids" or target == "3dpes" or target == "viper":
    #     cluster = DBSCAN(eps=0.3, min_samples=4, metric='precomputed', n_jobs=8)
    # elif target == "cuhk01":
    #     cluster = DBSCAN(eps=0.3, min_samples=4, metric='precomputed', n_jobs=8)
    # else:
    #     cluster = DBSCAN(eps=0.5, min_samples=4, metric='precomputed', n_jobs=8)
    # epsbytest__v3
    # if target == "ilids" or target == "3dpes" or target == "viper":
    #     cluster = DBSCAN(eps=0.3, min_samples=4, metric='precomputed', n_jobs=8)
    # elif target == "cuhk01":
    #     cluster = DBSCAN(eps=0.3, min_samples=4, metric='precomputed', n_jobs=8)
    # elif "cuhk03" in target:
    #     cluster = DBSCAN(eps=0.55, min_samples=4, metric='precomputed', n_jobs=8)
    # else:
    #     cluster = DBSCAN(eps=0.5, min_samples=4, metric='precomputed', n_jobs=8)
    # change mpt --> 4*weight
    mpt = 4*weight
    cluster = DBSCAN(eps=0.5, min_samples=4, metric='precomputed', n_jobs=8)
    updated_label = cluster.fit_predict(W)
    # updated_label = cluster.fit_predict(expended_W)


    # return expended_W, updated_label, new_pids
    return W, updated_label, new_pids

def print_mean_distance(dist_matrix, label, w):
    pos_dist_mean = 0
    hard_pos_mean = 0
    hard_neg_mean = 0
    for pid in np.unique(label):
        temp_dist_mean = 0
        temp_hard_pos_mean = 0
        temp_hard_neg_mean = 0
        if (pid != -1):
            idx = np.where(label == pid)[0]
            pos_len = round(0.1 * len(idx))
            neg_len = round(0.05 * (len(label)-len(idx)))
            for anchor_idx in idx:
                temp_dist_mean += dist_matrix[anchor_idx][idx].mean()
                # temp_dist_mean += np.sort(dist_matrix[anchor_idx][idx])[w:].mean()
                temp_hard_pos_mean += np.sort(dist_matrix[anchor_idx][idx])[-pos_len:].mean()
                temp_dist_matrix = np.delete(dist_matrix[anchor_idx], idx)
                temp_hard_neg_mean += np.sort(temp_dist_matrix)[:neg_len].mean()
            temp_dist_mean /= len(idx)
            temp_hard_pos_mean /= len(idx)
            temp_hard_neg_mean /= len(idx)
            pos_dist_mean += temp_dist_mean
            hard_pos_mean += temp_hard_pos_mean
            hard_neg_mean += temp_hard_neg_mean
        else:
            continue
            
    pos_dist_mean /= len(np.unique(label))
    hard_pos_mean /= len(np.unique(label))
    hard_neg_mean /= len(np.unique(label))

    return pos_dist_mean, hard_pos_mean, hard_neg_mean



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default="/home/tchsu/ICE_fed/examples/processed_data/")
    parser.add_argument("--datasets", nargs="+", default=["Duke","Market","ilids","cuhk03-np-detected","prid","viper","cuhk01","3dpes"])
    parser.add_argument("--target", type=str, default="ilids")
    parser.add_argument("--load_root", type=str, default="/home/tchsu/EasyFL_fix_logs")
    # parser.add_argument("--exp_name", type=str, default="best_dataset_append")
    # parser.add_argument("--exp_name", type=str, default="best_record_culster_score_save_localmodel")
    parser.add_argument("--exp_name", type=str, default="best_append_traintransform_tocluster")
    parser.add_argument("--save_txt", type=str, default="./evaluated_eps.txt")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f = open(args.save_txt, "a")
    f.write("\nappended dataset augtocluster\n")

    train_data = prepare_train_data(args.datasets, args.data_dir, "CAP")
    for target in args.datasets:
        # target = "cuhk03-np-detected"
        # load model
        # target = "Duke"
        model = stb_net.MemoryBankModel(out_dim=2048)
        state_dict = torch.load(f"{args.load_root}/{args.exp_name}/saved_model/{target}/best_lcoal_model.pth")
        model.load_state_dict(state_dict)
        model = model.to(device)

        # load data
        transform_val = T.Compose([
                    T.Resize(size=(256,128)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        transform_train = T.Compose([
                T.Resize((256,128)),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop((256,128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.RandomErasing(EPSILON=0.5)
                ])
        propagate_loader = DataLoader(
                UnsupervisedTargetPreprocessor(train_data.data[target],
                                                num_cam=train_data.dataset[target].num_cams, transform=transform_train),
                batch_size=32, num_workers=8,
                shuffle=False, pin_memory=True)
        
        # get weight for each dataset
        if target == "ilids":
            weight = 60
        elif target == "prid":
            weight = 4
        elif target == "viper":
            weight = 20
        elif target == "cuhk01":
            weight = 6
        elif target == "3dpes":
            weight = 30
        elif "cuhk03" in target:
            weight = 2
        else:
            weight = 1

        # main
        # k1 = 20 * (weight+1)
        # k2 = 6 * (weight+1)
        k1 = 20
        k2 = 6
        dist_matrix, p_label, gt_label= calculate_distance(model, propagate_loader, target, device, weight, k1=k1, k2=k2)
        f.write(target)
        f.write("\n")
        f.write('\tupdated_label: num_class= {}, {}/{} images are associated.'.format(p_label.max() + 1, len(p_label[p_label >= 0]), len(p_label)))  
        f.write("\n")
        s_score = silhouette_score(dist_matrix, p_label)
        h_score, c_score, v_score = homogeneity_completeness_v_measure(gt_label, p_label)
        f.write('\thomogeneity: {:.3f}, completeness: {:.3f}, vmeasure: {:.3f}, silhouette: {:.3f}'
                    .format(h_score, c_score, v_score, s_score))
        f.write("\n")
        gt_pos_dist, gt_hard_pos_mean, gt_hard_neg_mean = print_mean_distance(dist_matrix, gt_label, weight)
        f.write(f"\tpos_dist:{gt_pos_dist}\thard_pos_mean:{gt_hard_pos_mean}\thard_neg_mean:{gt_hard_neg_mean}")
        f.write("\n")

    f.close()