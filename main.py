import argparse
import logging
import easyfl

from client_cap import CAPClient
from server_cap import CAPServer
from dataset import prepare_train_data, prepare_test_data
from reid.models import stb_net



logger = logging.getLogger(__name__)


LOCAL_TEST = "local_test"
GLOBAL_TEST = "global_test"

RELABEL_LOCAL = "local"
RELABEL_GLOBAL = "global"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default="/home/tchsu/ICE_fed/examples/processed_data/")
    # parser.add_argument("--datasets", nargs="+", default=["ilids","viper","cuhk01","3dpes"])
    parser.add_argument("--datasets", nargs="+", default=["Duke","Market","ilids","cuhk03-np-detected","prid","viper","cuhk01","3dpes"])
    # parser.add_argument("--datasets", nargs="+", default=["ilids","viper","3dpes","prid","Duke","Market","cuhk01","cuhk03-np-detected"])
    parser.add_argument("--name", type=str, default="CAP")
    args = parser.parse_args()
    print("args:", args)

    # MAIN
    train_data = prepare_train_data(args.datasets, args.data_dir, args.name)
    test_data = prepare_test_data(args.datasets, args.data_dir, args.name)
    # Model
    if args.name == "CAP":
        model = stb_net.MemoryBankModel(out_dim=2048)
        easyfl.register_client(CAPClient)
        easyfl.register_server(CAPServer)
        config_file = "/home/tchsu/EasyFL_GTi2/config_CAP.yaml"
    # elif args.name == "O2CAP":
    #     model = models.create('resnet50', num_features=0, norm=True, dropout=0, num_classes=0, pool_type='avgpool')
    #     easyfl.register_client(O2CAPClient)
    #     config_file = "/home/remote/tchsu/EasyFL_update/config_O2CAP.yaml"
    else:
        pass

    easyfl.register_dataset(train_data, test_data)
    easyfl.register_model(model)
    
    # configurations
    # config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
    # config = easyfl.load_config(config_file, conf)    
    config = easyfl.load_config(config_file)

    print("config:", config)
    easyfl.init(config, init_all=True)
    easyfl.run()
