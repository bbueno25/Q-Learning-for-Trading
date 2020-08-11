"""
DOCSTRING
"""
import argparse
import collections
import gym
import itertools
import keras
import numpy
import pickle
import random
import re
import time

class DQNAgent:
    """
    A simple Deep Q agent.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = Model.mlp(state_size, action_size)

    def act(self, state):
        """
        Returns:
            action
        """
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return numpy.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """
        Vectorized implementation; 30x speed up compared with for loop.
        """
        minibatch = random.sample(self.memory, batch_size)
        states = numpy.array([tup[0][0] for tup in minibatch])
        actions = numpy.array([tup[1] for tup in minibatch])
        rewards = numpy.array([tup[2] for tup in minibatch])
        next_states = numpy.array([tup[3][0] for tup in minibatch])
        done = numpy.array([tup[4] for tup in minibatch])
        # Q(s', a)
        target = rewards + self.gamma * numpy.amax(self.model.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]
        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target
        self.model.fit(states, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

class TradingEnv(gym.Env):
    """
    A 3-stock (MSFT, IBM, QCOM) trading environment.

    State: [# of stock owned, current stock prices, cash in hand]
        - array of length n_stock * 2 + 1
        - price is discretized (to integer) to reduce state space
        - use close price for each stock
        - cash in hand is evaluated at each step based on action performed

    Action: sell (0), hold (1), and buy (2)
        - when selling, sell all the shares
        - when buying, buy as many as cash in hand allows
        - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """
    def __init__(self, train_data, init_invest=20000):
        # data
        self.stock_price_history = numpy.around(train_data) # round up to integer to reduce state space
        self.n_stock, self.n_step = self.stock_price_history.shape
        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        # action space
        self.action_space = gym.spaces.Discrete(3**self.n_stock)
        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        price_range = [[0, mx] for mx in stock_max_price]
        cash_in_hand_range = [[0, init_invest * 2]]
        self.observation_space = gym.spaces.MultiDiscrete(
            stock_range + price_range + cash_in_hand_range)
        # seed and start
        self._seed()
        self._reset()

    def _get_obs(self):
        obs = list()
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        return obs

    def _get_val(self):
        return numpy.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        return self._get_obs()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step] # update price
        self._trade(action)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(), reward, done, info

    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = map(list, itertools.product([0, 1, 2], repeat=self.n_stock))
        action_vec = action_combo[action]
        # one pass to get sell/buy index
        sell_index, buy_index = list(), list()
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)
        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False

class Model:
    """
    DOCSTRING
    """
    def mlp(
        self,
        n_obs,
        n_action,
        n_hidden_layer=1,
        n_neuron_per_layer=32,
        activation='relu',
        loss='mse'):
        """
        A multi-layer perceptron.
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
        for _ in range(n_hidden_layer):
            model.add(keras.layers.Dense(n_neuron_per_layer, activation=activation))
        model.add(keras.layers.Dense(n_action, activation='linear'))
        model.compile(loss=loss, optimizer=keras.optimizers.Adam())
        print(model.summary())
        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arser.add_argument(
        '-e', '--episode', type=int, default=2000, help='number of episode to run')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=32, help='batch size for experience replay')
    parser.add_argument(
        '-i', '--initial_invest', type=int, default=20000, help='initial investment amount')
    parser.add_argument(
        '-m', '--mode', type=str, required=True, help='either "train" or "test"')
    parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    args = parser.parse_args()
    Utils.maybe_make_dir('weights')
    Utils.maybe_make_dir('portfolio_val')
    timestamp = time.strftime('%Y%m%d%H%M')
    data = numpy.around(Utils.get_data())
    train_data = data[:, :3526]
    test_data = data[:, 3526:]
    env = TradingEnv(train_data, args.initial_invest)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scaler = Utils.get_scaler(env)
    portfolio_value = list()
    if args.mode == 'test':
        # remake the env with test data
        env = TradingEnv(test_data, args.initial_invest)
        # load trained weights
        agent.load(args.weights)
        # when test, the timestamp is same as time when weights was trained
        timestamp = re.findall(r'\d{12}', args.weights)[0]
    for e in range(args.episode):
        state = env.reset()
        state = scaler.transform([state])
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, episode end value: {}".format(
                    e + 1, args.episode, info['cur_val']))
            portfolio_value.append(info['cur_val']) # append episode end portfolio value
            break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        if args.mode == 'train' and (e + 1) % 10 == 0: # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))
    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)

class Utils:
    """
    DOCSTRING
    """
    def get_data(col='close'):
        """
        Return:
            a 3 x n_step array
        """
        msft = pd.read_csv('data/daily_MSFT.csv', usecols=[col])
        ibm = pd.read_csv('data/daily_IBM.csv', usecols=[col])
        qcom = pd.read_csv('data/daily_QCOM.csv', usecols=[col])
        # recent price are at top; reverse it
        return np.array(
            [msft[col].values[::-1], ibm[col].values[::-1], qcom[col].values[::-1]])

    def get_scaler(env):
        """
        Takes a env and returns a scaler for its observation space.
        """
        low = [0] * (env.n_stock * 2 + 1)
        high = list()
        max_price = env.stock_price_history.max(axis=1)
        min_price = env.stock_price_history.min(axis=1)
        max_cash = env.init_invest * 3 # 3 is a magic number
        max_stock_owned = max_cash // min_price
        for i in max_stock_owned:
            high.append(i)
        for i in max_price:
            high.append(i)
        high.append(max_cash)
        scaler = StandardScaler()
        scaler.fit([low, high])
        return scaler

    def maybe_make_dir(directory):
        """
        DOCSTRING
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
