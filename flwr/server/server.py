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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
#from actor_critic import Actor, Critic
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
#from flwr.server.env import SelectEnv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.pyplot import MultipleLocator

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

class SelectEnv:
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = []
        self.action = []
        self.state = []
        self.acc = 30 #use to control done's condition
        self.fit = None
        self.evaluate = None
        self.server = None
        self.count = 0
        
    def step(self, action):        
        self.action = self.action_space[action]
        # self.action = self.action_space[-1]
        print(f'\naction: {action, self.action}')
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
        
        self.count += 1
        log(INFO, f'training (COUNT: {self.count})')
        g_current_round += 1
       
        res_fit = self.server.fit_round(server_round=g_current_round, timeout=g_timeout)
        if res_fit:
            parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
            if parameters_prime:
                self.server.parameters = parameters_prime

        # Evaluate model on a sample of available clients
        log(INFO, "Evaluate 2")
        res_fed = self.server.evaluate_round(server_round=g_current_round, timeout=g_timeout)
        if res_fed:
            print(f'ACC: {res_fed[-1][-2][-1][-1].metrics["accuracy"]}')
            loss_fed, evaluate_metrics_fed, _ = res_fed
            acc = res_fed[-1][-2][-1][-1].metrics['accuracy']
            # if loss_fed:
            #     g_history.add_loss_distributed(
            #         server_round=g_current_round, loss=loss_fed
            #     )
            #     g_history.add_acc_distributed(
            #         server_round=g_current_round, acc=acc
            #     )
            #     g_history.add_metrics_distributed(
            #         server_round=g_current_round, metrics=evaluate_metrics_fed
            #     )

        # Calculate reward
        #if accuracy >= self.acc and self.state <=39: 
        #    reward =1 
        #else: 
        #    reward = -1 
        # print(f'loss: {loss_fed}')
        reward = acc * 100

        # Check if is done
        if reward >= self.acc: 
            print(f'ROUND: {g_current_round}, REWARD: {reward}')
            g_history.add_loss_distributed(
                server_round=g_current_round, loss=loss_fed
            )
            g_history.add_acc_distributed(
                server_round=g_current_round, acc=acc
            )
            # g_history.add_metrics_distributed(
            #     server_round=g_current_round, metrics=evaluate_metrics_fed
            # )
            if g_current_round >= g_num_round:
                done = True
            else:
                done = False

        else:
            g_current_round -= 1
            done = False
        
        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        states = self.state
        # states = self.server._client_manager.build_distance_matrix().flatten()
        # Return step information
        return states, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        #self.state = 38 + random.randint(-3,3)
        # Reset shower time
        #self.shower_length = 60 
        return self.state

    def build_model(self, states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,states)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model

    def build_agent(self, model, actions):
        policy = BoltzmannQPolicy()
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
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        global g_history
        g_history = History()

        global g_timeout
        global g_current_round
        global g_num_round

        g_timeout = timeout
        g_num_round = num_rounds

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
        client_num = len(self._client_manager)
        while len(self._client_manager) < 8:
            client_num = len(self._client_manager)
        print(client_num)

        fm = '{:0' + str(client_num) + 'b}'
        for i in range(pow(2, client_num)):
            b = fm.format(i)
            l = list(map(int, b))
            if l.count(1) >= client_num/3:
                self.env.action_space.append(l)
        
        print(self.env.action_space)
        self.env.action = self.env.action_space[-1]


        # Run federated learning for num_rounds
        log(INFO, "FL starting 0")
        start_time = timeit.default_timer()
        log(INFO, "FL starting round")

        for current_round in range(1, 3): #1, num_rounds + 1
            
            log(INFO, f'current_round: {current_round}')
            # Train model and replace previous global model

            #g_current_round = current_round

            log(INFO, "training")
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # #Evaluate model using strategy implementation
            # log(INFO, "Evaluate 1")
            # res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            # if res_cen is not None:
            #     loss_cen, metrics_cen = res_cen
            #     log(
            #         INFO,
            #         "fit progress: (%s, %s, %s, %s)",
            #         current_round,
            #         loss_cen,
            #         metrics_cen,
            #         timeit.default_timer() - start_time,
            #     )
            #     g_history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            #     g_history.add_metrics_centralized(
            #         server_round=current_round, metrics=metrics_cen
            #     )

            # # Evaluate model on a sample of available clients
            # log(INFO, "Evaluate 2")
            # res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            # if res_fed:
            #     loss_fed, evaluate_metrics_fed, _ = res_fed
            #     if loss_fed:
            #         g_history.add_loss_distributed(
            #             server_round=current_round, loss=loss_fed
            #         )
            #         g_history.add_metrics_distributed(
            #             server_round=current_round, metrics=evaluate_metrics_fed
            #         )

                
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

        self.env.server = self
        self.env.state = self._client_manager.build_distance_matrix().flatten()
        state_len = len(self.env.state)
        action_len = len(self.env.action_space)
        #print(np.shape(self.env.state))
        #print(state_len, action_len)
        fl_model = self.env.build_model(state_len, action_len)
        dqn = self.env.build_agent(fl_model, action_len)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.fit(self.env, nb_steps=num_rounds, visualize=False, verbose=1)
        print("SCORES")
        scores = dqn.test(self.env, nb_episodes=3, visualize=False)
        print(scores.history['episode_reward'])
        print(np.mean(scores.history['episode_reward']))



    ####################################################################################  

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)

        acc = g_history.acces_distributed
        plot_acc_loss([a for _, a in acc])

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
        #print(f'evaluate_round results: {results}')
        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        #print(list(aggregated_result))
        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

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
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


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

def plot_acc_loss(acc):
    plt.plot(acc)
    plt.title('Model accuracy')
    plt.xlabel('Round')
    plt.legend(['Train_acc'])
    x_major_locator=MultipleLocator(10)
    y_major_locator=MultipleLocator(0.1)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    #plt.xlim(-0.5,11)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0,1)
    #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.show()