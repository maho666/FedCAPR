import torch
import logging
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
# from sklearn.cluster.dbscan_ import dbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from reid.utils.rerank import compute_jaccard_dist
from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
import scipy.io as sio 
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)

def img_association(network, cap_memory, propagate_loader, device, min_sample=4, eps=0,
                    rerank=False, k1=20, k2=6):

    network.eval()
    # print('Start Inference...')
    features = []
    global_labels = []
    all_cams = []

    with torch.no_grad():
        for c, data in enumerate(propagate_loader):
            images = data[0].to(device) # data = (img, img1, img2, fname, pid, img_idx, camid)
            images1 = data[1].to(device) # data = (img, img1, img2, fname, pid, img_idx, camid)
            # images2 = data[2].to(device) # data = (img, img1, img2, fname, pid, img_idx, camid)
            g_label = data[4]
            cam = data[5]
            embed_feat = network(images)
            embed_feat1 = network(images1)
            # embed_feat2 = network(images2)
            concat_feat = torch.cat((embed_feat, embed_feat1), dim=1)
            features.append(concat_feat.cpu())

            global_labels.append(g_label)
            all_cams.append(cam)

    features = torch.cat(features, dim=0).numpy()
    global_labels = torch.cat(global_labels, dim=0).numpy()
    all_cams = torch.cat(all_cams, dim=0).numpy()
    # print('  features: shape= {}'.format(features.shape))
    # print('  global_labels: shape= {}'.format(global_labels.shape))
    # print('  all_cams: shape= {}'.format(all_cams.shape))

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

    # compute distance W
    new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # l2-normalize
    if rerank:
        W = faiss_compute_jaccard_dist(torch.from_numpy(new_features), k1=k1, k2=k2, print_flag=False)
    else:
        W = cdist(new_features, new_features, 'euclidean')
    # print('  distance matrix: shape= {}'.format(W.shape))

    # self-similarity for association
    # print('  perform image grouping...')
    # _, updated_label = DBSCAN(W, eps=eps, min_samples=min_sample, metric='precomputed', n_jobs=8)
    cluster = DBSCAN(eps=eps, min_samples=min_sample, metric='precomputed', n_jobs=8)
    updated_label = cluster.fit_predict(W)

    # print('  eps in cluster: {:.3f}'.format(eps))
    logger.info('updated_label: num_class= {}, {}/{} images are associated.'
          .format(updated_label.max() + 1, len(updated_label[updated_label >= 0]), len(updated_label)))

    intra_id_features = []
    # intra_id_labels = []
    proxy_label = -1*np.ones(updated_label.shape, updated_label.dtype)
    proxy_cnt = 0
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
                all_ind = np.array([percam_ind[i] for i in ind])
                proxy_label[all_ind] = proxy_cnt
                # intra_id_labels.append(lbl)
                cnt += 1
                proxy_cnt += 1
        percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
        intra_id_features.append(torch.from_numpy(percam_id_feature))

    proxy_label = torch.from_numpy(proxy_label)
    cap_memory.proxy_label_dict = {}
    for label in range(0, int(updated_label.max()+1)):
        cap_memory.proxy_label_dict[label] = torch.unique(proxy_label[updated_label == label])
    cap_memory.proxy_cam_dict = {}
    for cam in np.unique(new_cams): 
        proxy_inds = torch.unique(proxy_label[torch.from_numpy((new_cams == cam)) & (proxy_label >= 0)])
        cap_memory.proxy_cam_dict[int(cam)] = proxy_inds
        
    return updated_label, intra_id_features, proxy_label

