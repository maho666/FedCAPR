import torch
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
# from sklearn.cluster.dbscan_ import dbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from reid.utils.rerank import compute_jaccard_dist
from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
import scipy.io as sio 
import logging
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


def sup_img_association(network, propagate_loader, device, min_sample=4, eps=0,
                    rerank=False, k1=20, k2=6, intra_id_reinitialize=False):

    network.eval()
    # print('Start Inference...')
    features = []
    global_labels = []
    true_labels = []
    all_cams = []

    with torch.no_grad():
        for c, data in enumerate(propagate_loader):
            images = data[0].to(device)
            g_label = data[3]
            cam = data[4]
            t_label = data[2]
            embed_feat = network(images)
            features.append(embed_feat.cpu())
            true_labels.append(t_label)
            global_labels.append(g_label)
            all_cams.append(cam)

    features = torch.cat(features, dim=0).numpy()
    global_labels = torch.cat(global_labels, dim=0).numpy()
    all_cams = torch.cat(all_cams, dim=0).numpy()
    updated_label = torch.cat(true_labels, dim=0).numpy()

    # if needed, average camera-style transferred image features
    new_features = []
    new_cams = []
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        new_features.append(np.mean(features[idx], axis=0))
        new_cams.append(all_cams[idx])

    new_features = np.array(new_features)
    new_cams = np.array(new_cams).squeeze()
    del features, all_cams

    # # compute distance W
    # new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # l2-normalize
    # if rerank:
    #     W = faiss_compute_jaccard_dist(torch.from_numpy(new_features), k1=k1, k2=k2, print_flag=False)
    # else:
    #     W = cdist(new_features, new_features, 'euclidean')

    # # self-similarity for association
    # cluster = DBSCAN(eps=eps, min_samples=min_sample, metric='precomputed', n_jobs=8)
    # updated_label = cluster.fit_predict(W)

    

    # print('  eps in cluster: {:.3f}'.format(eps))
    logger.info('  updated_label: num_class= {}'.format(np.unique(updated_label).shape[0]))

    if intra_id_reinitialize:
        # print('re-computing initialized intra-ID feature...')
        intra_id_features = []
        intra_id_labels = []
        # print(new_cams)
        for cc in np.unique(new_cams):
            percam_ind = np.where(new_cams == cc)[0]
            percam_feature = new_features[percam_ind, :]
            percam_label = updated_label[percam_ind]
            percam_class_num = len(np.unique(percam_label[percam_label >= 0]))
            percam_id_feature = np.zeros((percam_class_num, percam_feature.shape[1]), dtype=np.float32)
            cnt = 0
            for lbl in np.unique(percam_label):
                if lbl >= 0:
                    ind = np.where(percam_label == lbl)[0]
                    id_feat = np.mean(percam_feature[ind], axis=0)
                    percam_id_feature[cnt, :] = id_feat
                    intra_id_labels.append(lbl)
                    cnt += 1
            percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
            intra_id_features.append(torch.from_numpy(percam_id_feature))
        return updated_label, intra_id_features

