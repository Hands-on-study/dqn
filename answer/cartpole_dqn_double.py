import os, sys, random, pylab
import numpy as np
from collections import deque

import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

EPISODES = 300

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        self.memory = deque(maxlen=2000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn_double_trained.h5")

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))  # ← input_dim 경고 제거
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # ← lr -> learning_rate
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_value = self.model.predict(state, verbose=0)  # ← verbose=0로 깔끔하게
        return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states, verbose=0)


        # 1. 행동 '선택'을 위해 메인 네트워크로 다음 상태의 Q-값 예측
        next_q_values_main = self.model.predict(next_states, verbose=0)

        # 2. 가치 '평가'를 위해 타겟 네트워크로 다음 상태의 Q-값 예측
        target_val = self.target_model.predict(next_states, verbose=0)


        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Step 1: 메인 네트워크의 예측값으로 최적 행동의 '인덱스'를 선택
                action = np.argmax(next_q_values_main[i])

                # Step 2: 타겟 네트워크의 예측값에서 위에서 선택한 행동의 '가치'를 평가
                value = target_val[i][action]
                
                # Step 3: 위에서 평가한 가치(value)를 이용해 최종 타겟 계산
                target[i][actions[i]] = rewards[i] + self.discount_factor * value

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":
    os.makedirs("./save_graph", exist_ok=True)
    os.makedirs("./save_model", exist_ok=True)

    env = gym.make('CartPole-v1', render_mode="human")  # Gymnasium은 기본적으로 v1 OK
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        # Gymnasium reset: (obs, info)
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size]).astype(np.float32)

        done = False
        score = 0

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)

            # Gymnasium step: (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)

            # 실패 시 -100 보상
            shaped_reward = reward if not done or score == 499 else -100.0

            agent.append_sample(state, action, shaped_reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                agent.update_target_model()

                final_score = score if score == 500 else score + 100
                scores.append(final_score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn_double.png")

                print(f"episode: {e}  score: {final_score:.1f}  memory length: {len(agent.memory)}  epsilon: {agent.epsilon:.4f}")

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/cartpole_dqn_double.weights.h5")
                    sys.exit()
