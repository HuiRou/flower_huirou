# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower ClientManager."""


import random
import threading
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import GetParametersIns

from flwr.common.logger import log

from .client_proxy import ClientProxy
from .criterion import Criterion

from sklearn.metrics import pairwise_distances

from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

class ClientManager(ABC):
    """Abstract base class for managing Flower clients."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients."""

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful
        """

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance."""

    @abstractmethod
    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""

    @abstractmethod
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Wait until at least `num_clients` are available."""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        selection: Optional[list] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""


    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self.clients)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Block until at least `num_clients` are available or until a timeout
        is reached.

        Current timeout default: 1 day.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def num_available(self) -> int:
        """Return the number of available clients."""
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful. False if ClientProxy is
                already registered or can not be registered for any reason
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        #self.client_weight[client.cid] = None
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        selection: Optional[list] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        
        sampled_cids = random.sample(available_cids, num_clients)
        #sampled_cids = available_cids

        return [self.clients[cid] for cid in sampled_cids]

    def sample_selection(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        selection: Optional[list] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        # if num_clients > len(available_cids):
        #     log(
        #         INFO,
        #         "Sampling failed: number of available clients"
        #         " (%s) is less than number of requested clients (%s).",
        #         len(available_cids),
        #         num_clients,
        #     )
        #     return []
        if(len(selection) != len(available_cids)):
            log(INFO, "len(selection) != len(available_cids)")   
        sampled_cids = [available_cids[i] for i in range(len(selection)) if selection[i] == 1]
        #sampled_cids = random.sample(available_cids, num_clients)
        #sampled_cids = available_cids

        return [self.clients[cid] for cid in sampled_cids]

    def build_distance_matrix(self):
        log(INFO, "build_distance_matrix")   
        # print(self.clients)     
        print(self.clients.keys())
        ins = GetParametersIns(config={})
        weight = []
        weights = []
        result_list = []
        client_weight = {}
        for cid in self.clients.keys():
            res = self.clients[cid].get_parameters(ins=ins, timeout=None)
            weight = parameters_to_ndarrays(res.parameters)
            result_list = []
            for i in range(len(weight)):
                result = weight[i].flatten() # to 1D
                result_list.extend(result)
            weights.append(result_list)
        distance_matrix = pairwise_distances(weights, metric='euclidean')
        print(distance_matrix)
        return distance_matrix


    def clustering(self):
        log(INFO, "Clustering")        
        ins = GetParametersIns(config={})
        weight = []
        weights = []
        gradients = []
        client_list = []
        for client in self.clients.values(): #client -> ClientProxy
            log(INFO, "Requesting parameters from client: " + str(client.cid))
            client_list.append(client.cid[-5:])
            res = client.get_parameters(ins=ins, timeout=None)
            weight = parameters_to_ndarrays(res.parameters)
            result_list = []
            for i in range(len(weight)):
                result = weight[i].flatten() # to 1D
                result_list.extend(result)
            
            # Gradient
            x = tf.Variable(result_list)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                y = tf.nn.softplus(x)
                #y = x**2
            #print(g.gradient(y, x))
            gradients.append(g.gradient(y, x))

            weights.append(result_list)
        print(np.shape(weights))

        # Clustering
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_distances=True).fit(gradients)
        #cluster = AgglomerativeClustering(n_clusters=2, compute_distances=True).fit(weights)
        #cluster = KMeans(n_clusters=2, n_init='auto').fit(weights)
        labels = cluster.labels_
        print(labels)

        #print(cluster.n_leaves_)
        #print(cluster.distances_)

        print(cluster.n_leaves_)
        print(cluster.distances_)

        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        plot_dendrogram(cluster, truncate_mode='level', p=10)
        labels = [item.get_text() for item in plt.gca().get_xticklabels()]
        #print(labels)
        new_labels = []
        new_labels = [client_list[int(x)] for x in labels]

        plt.gca().set_xticklabels(new_labels)

        plt.show()
        