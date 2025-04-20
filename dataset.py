import os
import sys

from easyfl.datasets import FederatedImageDataset
from easyfl.datasets.simulation import SIMULATE_IID
from reid.datasets import DS as CAP_DS
from reid.datasets import old_DS as mine_DS
# from reid_o2cap.datasets import DS as O2CAP_DS

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def prepare_train_data(db_names, data_dir, name):
    client_ids = []
    roots = []
    for d in db_names:
        client_ids.append(d)
        if (d == "Duke") or (d == "Market"):
            data_path = os.path.join(data_dir, 'FedUReID', d, 'pytorch')
            roots.append(os.path.join(data_path, 'train_all'))
        else:
            data_path = os.path.join(data_dir, d, 'pytorch')
            roots.append(os.path.join(data_path, 'train_all'))
    data = MyImageDataset(root=roots,
                            simulated=True,
                            do_simulate=False,
                            client_ids=client_ids,
                            name=name)
    return data


def prepare_test_data(db_names, data_dir, name):
    roots = []
    client_ids = []
    for d in db_names:
        client_ids.extend(["{}_{}".format(d, "gallery"), "{}_{}".format(d, "query"), "{}_{}".format(d, "trainset")])
        if (d == "Duke") or (d == "Market"):
            test_gallery = os.path.join(data_dir, 'FedUReID', d, 'pytorch', 'gallery')
            test_query = os.path.join(data_dir, 'FedUReID', d, 'pytorch', 'query')
            test_train = os.path.join(data_dir, 'FedUReID', d, 'pytorch', 'train_all')
            roots.extend([test_gallery, test_query, test_train])
        else:
            test_gallery = os.path.join(data_dir, d, 'pytorch', 'gallery')
            test_query = os.path.join(data_dir, d, 'pytorch', 'query')
            test_train = os.path.join(data_dir, d, 'pytorch', 'train_all')
            roots.extend([test_gallery, test_query, test_train])
        # client_ids.extend(["{}_{}".format(d, "gallery"), "{}_{}".format(d, "query")])
        # if (d == "Duke") or (d == "Market"):
        #     test_gallery = os.path.join(data_dir, 'FedUReID', d, 'pytorch', 'gallery')
        #     test_query = os.path.join(data_dir, 'FedUReID', d, 'pytorch', 'query')
        #     roots.extend([test_gallery, test_query])
        # else:
        #     test_gallery = os.path.join(data_dir, d, 'pytorch', 'gallery')
        #     test_query = os.path.join(data_dir, d, 'pytorch', 'query')
        #     roots.extend([test_gallery, test_query])
        
    data = MyImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 client_ids=client_ids,
                                 name=name)
    return data

class MyImageDataset(FederatedImageDataset):
    def __init__(self,root,simulated,do_simulate=True,extensions=IMG_EXTENSIONS,is_valid_file=None,target_transform=None,
                 client_ids="default",num_of_clients=10,simulation_method=SIMULATE_IID,weights=None,alpha=0.5,min_size=10,
                 class_per_client=1,name='default'):
        self.name = name
        super(MyImageDataset, self).__init__(root,simulated,do_simulate,extensions,is_valid_file,target_transform,client_ids,
                                             num_of_clients,simulation_method,weights,alpha,min_size,class_per_client)
        

    def make_DS(self):
        # if self.name == "CAP": 
        if self.client_ids == "default":
            self.users = ["f%07.0f" % (i) for i in range(len(self.roots))]
        else:
            self.users = self.client_ids
        for i in range(self.num_of_clients):
            current_client_id = self.users[i]
            self.dataset[current_client_id] = CAP_DS(current_client_id, self.roots[i])
            self.data[current_client_id] = self.dataset[current_client_id].data
        # elif self.name == "O2CAP": 
        #     if self.client_ids == "default":
        #         self.users = ["f%07.0f" % (i) for i in range(len(self.roots))]
        #     else:
        #         self.users = self.client_ids
        #     for i in range(self.num_of_clients):
        #         current_client_id = self.users[i]
        #         self.dataset[current_client_id] = O2CAP_DS(current_client_id, self.roots[i])
        #         self.data[current_client_id] = self.dataset[current_client_id].data
        # elif self.name == "mine_DS":  #### CAP + mineDS ####
        #     root = "/home/remote/tchsu/ICE_fed/examples/data/"
        #     if self.client_ids == "default":
        #         self.users = ["f%07.0f" % (i) for i in range(len(self.roots))]
        #     else:
        #         self.users = self.client_ids
        #     for i in range(self.num_of_clients):
        #         current_client_id = self.users[i]
        #         self.dataset[current_client_id] = mine_DS(current_client_id, root)
        #         self.data[current_client_id] = self.dataset[current_client_id].data
        # else:
        #     pass

    # def make_DS(self):
    #     if self.client_ids == "default":
    #         self.users = ["f%07.0f" % (i) for i in range(len(self.roots))]
    #     else:
    #         self.users = self.client_ids
    #     for i in range(self.num_of_clients):
    #         current_client_id = self.users[i]
    #         self.dataset[current_client_id] = DS(current_client_id, self.roots[i])
    #         self.data[current_client_id] = self.dataset[current_client_id].data

