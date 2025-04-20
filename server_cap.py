from easyfl.server.base import BaseServer
from easyfl.protocol import codec
from easyfl.tracking import metric
from easyfl.utils.float import rounding
from omegaconf import OmegaConf
import torch
import time

# train and test params
MODEL = "model"
DATA_SIZE = "data_size"
ACCURACY = "accuracy"
LOSS = "loss"
CLIENT_METRICS = "client_metrics"

FEDERATED_AVERAGE = "FedAvg" 

AGGREGATION_CONTENT_ALL = "all"
AGGREGATION_CONTENT_PARAMS = "parameters"

class CAPServer(BaseServer):
    def __init__(self,conf,test_data=None,val_data=None,is_remote=False,local_port=22999):
        super(CAPServer, self).__init__(conf, test_data, val_data,is_remote,local_port)

    def start(self, model, clients):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
                Clients are actually client grpc addresses when in remote training.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        self.set_model(model)
        self.set_clients(clients)

        if self._should_track():
            self._tracker.create_task(self.conf.task_id, OmegaConf.to_container(self.conf))

        # Get initial testing accuracies
        if self.conf.server.test_all:
            if self._should_track():
                self._tracker.set_round(self._current_round)
            self.test()
            self.save_tracker()

        # Setup global memroy bank
        self.global_memory_bank = []

        while not self.should_stop():
            self._round_time = time.time()

            self._current_round += 1
            self.print_("\n-------- round {} --------".format(self._current_round))

            # Train
            self.pre_train()
            self.train()
            self.post_train()

            # Test
            if self._do_every(self.conf.server.test_every, self._current_round, self.conf.server.rounds):
                self.pre_test()
                self.test()
                self.post_test()

            # Save Model
            self.save_model()

            self.track(metric.ROUND_TIME, time.time() - self._round_time)
            self.save_tracker()

        self.print_("Accuracies: {}".format(rounding(self._accuracies, 4)))
        self.print_("Cumulative training time: {}".format(rounding(self._cumulative_times, 2)))


    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        client_memory_bank = []
        for i, client in enumerate(self.grouped_clients):
            # Update client config before training
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round

            # uploaded_request, weight = client.run_train(self._compressed_model, self.conf.client)
            uploaded_request, upload_memory_bank, weight = client.run_train(self._compressed_model, self.conf.client, self.global_memory_bank, i)
            uploaded_content = uploaded_request.content

            client_memory_bank.append(upload_memory_bank)
            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model
            # uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_weights[client.cid] = weight
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)
        self.global_memory_bank = client_memory_bank

    def set_client_uploads_train(self, models, weights, metrics=None):
        """Set training updates uploaded from clients.

        Args:
            models (dict): A collection of models.
            weights (dict): A collection of weights.
            metrics (dict): Client training metrics.
        """
        self.set_client_uploads(MODEL, models)
        self.set_client_uploads(DATA_SIZE, weights)
        if self._should_gather_metrics():
            metrics = self.gather_client_train_metrics()
        self.set_client_uploads(CLIENT_METRICS, metrics)

    def aggregation(self):
        """Aggregate training updates from clients.
        Server aggregates trained models from clients via federated averaging.
        """
        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())

        model = self.aggregate(models, weights)
        self.set_model(model, load_dict=True)

    

