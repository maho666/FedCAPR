from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import argparse
from matplotlib.pyplot import cm
import numpy as np
import torch
import math
import os
from torch.utils.data import DataLoader

from reid.models import stb_net
from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
from reid.utils.data.preprocessor import UnsupervisedTargetPreprocessor
from reid.utils.data import transforms as T
from dataset import prepare_train_data, prepare_test_data

def get_data_loader(train_data, name):
    transform_val = T.Compose([
                T.Resize(size=(256,128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    data_loader =  DataLoader(
        UnsupervisedTargetPreprocessor(train_data.data[name],
                                        num_cam=train_data.dataset[name].num_cams, 
                                        transform=transform_val),
        batch_size=32, num_workers=8, shuffle=False, pin_memory=True)

    return data_loader

def get_feature(model, data_loaders, device):
    model.to(device)
    model.eval()
    features = []
    g_labels = []
    ds_labels = []
    pid_container = set()

    with torch.no_grad():
        for i, data_loader in enumerate(data_loaders):
            for c, data in enumerate(data_loader):
                images = data[0].to(device)
                label = data[2]
                feat = model(images)

                features.append(feat.cpu())
                g_labels.append(label)

                lenth = label.shape[0]
                ds_labels.append(torch.tensor([i for num in range(lenth)]))

    features = torch.cat(features, dim=0).numpy()
    g_labels = torch.cat(g_labels, dim=0).numpy()
    ds_labels = torch.cat(ds_labels, dim=0).numpy()

    for pid in g_labels:
        pid_container.add(pid)

    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    for i, pid in enumerate(g_labels):
        g_labels[i] = pid2label[pid]

    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    W = faiss_compute_jaccard_dist(torch.from_numpy(features), k1=20, k2=6, print_flag=False)
    cluster = DBSCAN(eps=0.5, min_samples=4, metric='precomputed', n_jobs=8)
    updated_label = cluster.fit_predict(W)

    return W, updated_label, g_labels, ds_labels


 
def tsne(args, features, labels, cid):
    tsne = TSNE(n_components=2, metric="precomputed")
    result = tsne.fit_transform(features)
    x_min, x_max = result.min(0), result.max(0)
    norm = (result - x_min) / (x_max - x_min)

    if cid == "global":
        title = f"{cid}_{args.exp}_ds36"
    else:
        title = f"{cid}_{args.exp}"

    save_path = os.path.join(args.save_root, title+".png")

    id_num = np.unique(np.array(labels)).shape[0]
    color = cm.rainbow(np.linspace(0,1,id_num))
    fig = plt.figure(dpi=300)
    for i in range(norm.shape[0]):
        plt.text(norm[i, 0], norm[i, 1], str(labels[i]), color=color[labels[i]], fontsize="xx-small")
    plt.title(title)

    fig.savefig(save_path)
    

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default="/home/tchsu/ICE_fed/examples/processed_data/")
    # parser.add_argument("--datasets", nargs="+", default=["Duke","Market","ilids","cuhk03-np-detected","prid","viper","cuhk01","3dpes"])
    parser.add_argument("--datasets", nargs="+", default=["cuhk03-np-detected","cuhk01"])
    parser.add_argument("--name", type=str, default="CAP")
    parser.add_argument("--exp", type=str, default="2stage_local2global")
    parser.add_argument("--round_num", type=str, default="19")
    parser.add_argument("--task", type=str, default="global")
    parser.add_argument("--save_root", type=str, default="/home/tchsu/EasyFL_update_logs/tsne")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model_path = f'/home/tchsu/EasyFL_update_logs/{args.exp}/saved_model/{args.exp}_global_model_r_{args.round_num}.pth'
    model = stb_net.MemoryBankModel(out_dim=2048)
    model.load_state_dict(torch.load(model_path))

    train_data = prepare_train_data(args.datasets, args.data_dir, args.name)

    if args.task == "global":
        loaders = []
        for cid in args.datasets:
            loader = get_data_loader(train_data, cid)
            loaders.append(loader)
        dist, cluster_labels, g_labels, ds_labels = get_feature(model, loaders, device)
        tsne(args, dist, ds_labels, "global")

    elif args.task == "local":
        for cid in args.datasets:
            loaders = []
            data_loader = get_data_loader(train_data, cid)
            loaders.append(data_loader)
            dist, cluster_labels, g_labels, ds_labels = get_feature(model, loaders, device)
            tsne(args, dist, g_labels, cid)
    else:
        pass






