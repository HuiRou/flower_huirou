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
"""Flower server."""


import concurrent.futures
import timeit
import datetime
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
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
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.common.typing import ResetIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
#from actor_critic import Actor, Critic
from rl.agents import DQNAgent
from rl.policy import (
    BoltzmannQPolicy,
    LinearAnnealedPolicy,
    SoftmaxPolicy,
    EpsGreedyQPolicy,
    GreedyQPolicy,
    MaxBoltzmannQPolicy,
)
from rl.memory import SequentialMemory
#from flwr.server.env import SelectEnv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.pyplot import MultipleLocator
import copy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


g_timeout = 0
g_current_round = 0
g_num_round = 0
g_history = None
g_mode = None
g_step = 0
g_episode = 0

class SelectEnv:
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = None
        self.action = None
        self.init_state = None
        self.state = None
        self.matrix = None
        self.server = None
        self.count = 0
        self.init_acc = 0
        self.acc = 0 # use to control done's condition
        
    def step(self, action):        
        self.action = self.action_space[action]
        # self.action = self.action_space[-1]
        print(f'\naction: {action, self.action}')

        #FEDAVG/RD
        #agg_models = [models[i] for i in range(len(actions)) if actions[i] == 1]
        #parameters_aggregated = aggregate(agg_models)     
        #global global_model   
        #global_model.set_weights(parameters_aggregated)
        #(x_train, y_train), (x_test, y_test) = data_load(999)
        #loss, accuracy = global_model.evaluate(x_test, y_test)

        #print(f'Aggregate, loss: {loss}, accuracy: {accuracy}')
        
        global g_history
        global g_timeout
        global g_current_round
        global g_num_round
        global g_mode
        global g_step
        global g_episode
        #self.count += 1
        log(INFO, f'training (ROUND: {g_current_round})')
       
        res_fit = self.server.fit_round(server_round=g_current_round, timeout=g_timeout)
        if res_fit:
            parameters_prime, metric_fed, _ = res_fit  # fit_metrics_aggregated
            if parameters_prime:
                self.server.parameters = parameters_prime
            #acc = metric_fed['accuracy']

        # Evaluate model on a sample of available clients
        log(INFO, "Evaluate 2")
        res_fed = self.server.evaluate_round(server_round=g_current_round, timeout=g_timeout)
        if res_fed:
            #print(f'ACC: {res_fed[-1][-2][-1][-1].metrics["accuracy"]}')
            loss_fed, acc_fed, evaluate_metrics_fed, _ = res_fed
            print(f'ACC:{acc_fed}')
            acc = acc_fed

            g_history.add_acc_distributed(
                server_round=g_current_round, acc=acc
            )
            g_history.add_cnt_distributed(
                server_round=g_current_round, cnt=self.action.count(1)
            )
            a=copy.deepcopy(self.action)
            a.append(action)
            if g_mode == 'train':                
                g_history.add_train_action_distributed(
                    server_round=g_current_round, action=a
                )
            if g_mode == 'test':
                g_history.add_test_action_distributed(
                    server_round=g_current_round, action=a
                )
                
                ##########       ####          
                ###########      ####
                ####    ####     ####
                ####    ####     ####
    ########    ###########      ####            ########
    ########    ##########       ####            ########
                ####    ####     ####
                ####    ####     ####
                ####    ####     ############
                ####    ####     ############

        #reward = (acc-self.acc)*100 - 100
        reward = acc*100 - self.action.count(1) # client num=8

        states = []
        
        print(f'Server: r={g_current_round}, e={g_episode}, s={g_step}, acc={acc}, pa={self.acc}, rw={reward}')
            
        # Check if is done ######################################################################################
        if g_mode == "test":
            condition = 1
        else:
            condition = 0.5

        if acc > condition:        
            print("done!")
            g_history.add_ep_distributed(
                server_round=(g_current_round), ep=g_episode
            )
            # g_current_round += 1
            done = True
        else:
            done = False

        if done:
            g_step = 0
            g_episode += 1
            #self.acc = self.init_acc
            self.acc = acc
        else:
            g_step += 1
            self.acc = acc

        info = {}

        g_current_round += 1
        for m in self.matrix:
            s = [m[i] for i in range(len(self.matrix)) if self.action[i] == 1]
            states.append(np.mean(s))
        states.append(acc)
        print(f'NEXT_STATE = {states}')
        return states, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        global g_timeout
        self.server.reset(timeout=g_timeout)
        #print(f'now acc: {self.acc} / init_acc: {self.init_acc}')

        self.acc = self.init_acc
        self.state = self.init_state
        return self.state

    def build_model(self, states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,states)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='softmax'))
        return model

    def build_agent(self, model, actions):
        policy = BoltzmannQPolicy()
        # policy = EpsGreedyQPolicy()
        # policy = GreedyQPolicy()
        # policy = MaxBoltzmannQPolicy()

        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
        return dqn

class Server:
    """Flower server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.env = SelectEnv()
    
    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, mode: str, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        global g_history
        g_history = History()

        global g_timeout
        global g_current_round
        global g_num_round
        global g_mode 

        g_timeout = timeout
        g_num_round = num_rounds
        g_mode = mode

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            #g_history.add_loss_centralized(server_round=0, loss=res[0])
            #g_history.add_metrics_centralized(server_round=0, metrics=res[1])
        
        while len(self._client_manager) < 8:
            continue
        client_num = len(self._client_manager)
        print(client_num)
        action_space = []
        fm = '{:0' + str(client_num) + 'b}'
        for i in range(pow(2, client_num)):
            b = fm.format(i)
            l = list(map(int, b))
            if l.count(1) > 0: #  >= client_num/3
                action_space.append(l)
        self.env.action_space = action_space
        print(len(self.env.action_space))
        #print(self.env.action_space)
        self.env.action = self.env.action_space[-1]


        # Run federated learning for num_rounds
        log(INFO, "FL starting 0")
        start_time = timeit.default_timer()
        log(INFO, "FL starting round")

                    ############    ####          
                    ############    ####
                    ####            ####
                    ####            ####
        ########    ############    ####            ########
        ########    ############    ####            ########
                    ####            ####
                    ####            ####
                    ####            ############
                    ####            ############


        if g_mode == "avg":
            print("AVG")
            for current_round in range(0, num_rounds): #1, num_rounds + 1
            
                log(INFO, f'current_round: {current_round}')
                # Train model and replace previous global model

                #g_current_round = current_round
                
                log(INFO, "training")
                res_fit = self.fit_round(server_round=current_round, timeout=timeout)
                if res_fit:
                    parameters_prime, metric_fed, _ = res_fit  # fit_metrics_aggregated
                    if parameters_prime:
                        self.parameters = parameters_prime
                   # acc = metric_fed['accuracy']
       

                log(INFO, "Evaluate 2")
                res_fed = self.evaluate_round(server_round=g_current_round, timeout=timeout)
                if res_fed:
                    #print(f'ACC: {res_fed[-1][-2][-1][-1].metrics["accuracy"]}')
                    loss_fed, acc_fed, evaluate_metrics_fed, _ = res_fed
                    #acc = res_fed[-1][-2][-1][-1].metrics['accuracy']
                    acc = acc_fed
                    g_history.add_acc_distributed(
                    server_round=g_current_round, acc=acc
                    )
                print(f'Server: r={current_round}, acc={acc}')
        elif g_mode == 'random':
            print("RD")
            for current_round in range(0, num_rounds): #1, num_rounds + 1
            
                log(INFO, f'current_round: {current_round}')
                # Train model and replace previous global model

                #g_current_round = current_round
                
                log(INFO, "training")
                res_fit = self.fit_round(server_round=current_round, timeout=timeout)
                if res_fit:
                    parameters_prime, metric_fed, _ = res_fit  # fit_metrics_aggregated
                    if parameters_prime:
                        self.parameters = parameters_prime
                   # acc = metric_fed['accuracy']
                    g_history.add_cnt_distributed(
                        server_round=current_round, cnt=metric_fed['cnt']
                    )

                log(INFO, "Evaluate 2")
                res_fed = self.evaluate_round(server_round=g_current_round, timeout=timeout)
                if res_fed:
                    #print(f'ACC: {res_fed[-1][-2][-1][-1].metrics["accuracy"]}')
                    loss_fed, acc_fed, evaluate_metrics_fed, _ = res_fed
                    #acc = res_fed[-1][-2][-1][-1].metrics['accuracy']
                    acc = acc_fed
                    g_history.add_acc_distributed(
                    server_round=g_current_round, acc=acc
                    )
                print(f'Server: r={current_round}, acc={acc}')
        else:
            for current_round in range(0, 2): #1, num_rounds + 1
                print(self.env.action)
                log(INFO, f'current_round: {current_round}')
                # Train model and replace previous global model

                #g_current_round = current_round
                
                log(INFO, "training")
                res_fit = self.fit_round(server_round=current_round, timeout=timeout)
                if res_fit:
                    parameters_prime, metric_fed, (result, failure) = res_fit  # fit_metrics_aggregated
                    if parameters_prime:
                        self.parameters = parameters_prime      
            
            self._client_manager.order = {
                client_proxy.cid:fit_res.metrics['id']
                for client_proxy, fit_res in result
                }
            print(f'Order_ori:\n\t{self._client_manager.order}')
            self._client_manager.order = dict(sorted(self._client_manager.order.items(), key=lambda x:x[1]))
            self._client_manager.clients = {cid:self._client_manager.clients[cid] for cid in self._client_manager.order.keys()}
            print(f'Clients:\n\t{self._client_manager.clients.keys()}')
            self.env.matrix = self._client_manager.build_distance_matrix()

            log(INFO, "Evaluate 2")
            res_fed = self.evaluate_round(server_round=g_current_round, timeout=timeout)
            if res_fed:
                #print(f'ACC: {res_fed[-1][-2][-1][-1].metrics["accuracy"]}')
                loss_fed, acc_fed, evaluate_metrics_fed, _ = res_fed
                #acc = res_fed[-1][-2][-1][-1].metrics['accuracy']
                acc = acc_fed
                self.env.init_acc = acc 
                self.env.acc = acc
                print(f'env.init_acc: {self.env.init_acc}')
        
            
            self.env.server = self
            print(self.env.matrix)
            state = []
            for m in self.env.matrix:
                state.append(np.mean(m))
            state.append(acc)
            self.env.init_state = state

            state_len = len(self.env.init_state)
            action_len = len(self.env.action_space)
            #print(np.shape(self.env.state))
            #print(state_len, action_len)
            fl_model = self.env.build_model(state_len, action_len)
            dqn = self.env.build_agent(fl_model, action_len)
            dqn.compile(Adam(lr=1e-3), metrics=['mae'])

            if g_mode == "train":
                print("TRAIN")
                dqn.fit(self.env, nb_steps=num_rounds, action_repetition=1, visualize=False, verbose=1)
                dt = datetime.datetime.now().strftime("%m%d_%H%M")
                fn = f'{dt}/dqn_weights.h5f'
                dqn.save_weights(fn, overwrite=True)
                print(f"SAVE!!! {fn}")

            
                # dt = datetime.datetime.now().strftime("%m%d_%H%M")

                        ############    ############    ############    ############
                        ############    ############    ############    ############
                            ####        ####            ####                ####
                            ####        ####            ####                ####
            ########        ####        ############    ############        ####    ########
            ########        ####        ############    ############        ####    ########
                            ####        ####                    ####        ####    
                            ####        ####                    ####        ####    
                            ####        ############    ############        ####    
                            ####        ############    ############        ####    

            if g_mode == "test":
                print("TEST")
                #fn = f'{dt}/dqn_weights.h5f'
                fn = f'0817_1516/dqn_weights.h5f'
                dqn.load_weights(fn)

                scores = dqn.test(self.env, nb_episodes=1, visualize=False,nb_max_episode_steps=100)
                print(f'SCORES: {scores.history["episode_reward"]}')
                print(f'SCORE MEAN: {np.mean(scores.history["episode_reward"])}')
            
        
    ####################################################################################  

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)

        acc_his = g_history.acces_distributed #[a for _, a in acc]
        ep_his = g_history.epes_distributed
        
        log(INFO, "app_fit: acces_distributed\n%s", str(g_history.acces_distributed))
        log(INFO, "app_fit: cntes_distributed\n%s", str(g_history.cntes_distributed))
        log(INFO, "app_fit: epes_distributed\n%s", str(g_history.epes_distributed))
        log(INFO, "app_fit: train_actions_distributed\n%s", str(g_history.train_actions_distributed))
        log(INFO, "app_fit: test_actions_distributed\n%s", str(g_history.test_actions_distributed))

        plot_acc_loss(acc_his, ep_his)

        return g_history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        # print(f'evaluate_round results: {results}')
        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        # print(list(aggregated_result))
        loss_aggregated, acc_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, acc_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
            selection=self.env.action,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        metrics_aggregated['cnt'] = len(client_instructions)
        return parameters_aggregated, metrics_aggregated, (results, failures)    

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={'reset': True})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters

    def reset(self, timeout: Optional[float]):
        """Get initial parameters from one of the available clients."""

        log(INFO, "Reset client Parameter")
        #clients = self._client_manager.sample(8)[0]
        ins = GetParametersIns(config={'reset': True})
        # ins = ResetIns(config={})
        clients = self._client_manager.clients
        for cid in clients.keys():
            get_parameters_res = clients[cid].get_parameters(ins=ins, timeout=timeout)

######## main ########

def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect

def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, __ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures

def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res

def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    #print(f'evaluate_res: {evaluate_res}')
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def plot_acc_loss(acc_list, ep_list):
    global g_mode
    plt.style.use('bmh')
    fig = plt.figure()
    ax = plt.axes() #畫上刻度

    acc = [a for _, a in acc_list]    

    if len(ep_list) > 0:
        round =  [r for r, _ in ep_list]
        r_a =  [acc[r] for r, _ in ep_list]
        plt.plot(round, r_a, color="red", marker=".", markersize=7,label="reset", linestyle="None")
    plt.plot(acc, label="acc")

    dt = datetime.datetime.now().strftime("%m/%d %H:%M")

    plt.title(f'Model accuracy {g_mode} {dt}')
    plt.xlabel('Round')
    plt.legend()

    x_major_locator=MultipleLocator(10)
    y_major_locator=MultipleLocator(0.1)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    #plt.xlim(-5,400)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-0.05,1)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.show()