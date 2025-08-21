## Quiz
1. 리플레이 버퍼에서 학습에 사용할 미니배치를 `random.sample`을 이용해 무작위로 추출하는 주된 이유는 무엇일까?

    ```python
    # train_model 함수 중
    mini_batch = random.sample(self.memory, self.batch_size)
    ```
    A. 가장 보상이 높았던 경험들만 골라서 학습하기 위해  
    B. 가장 최근에 수집된 경험을 우선적으로 학습하기 위해  
    C. 데이터 샘플 간의 시간적 상관관계를 제거하여 학습을 안정시키기 위해  
    D. 리플레이 버퍼의 메모리 사용량을 줄이기 위해  

    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <b>C. 강화학습에서 데이터는 시간 순서대로 수집되므로 바로 인접한 샘플들은 매우 유사하다. 이 데이터들을 순서대로 학습하면 모델이 특정 패턴에 과적합될 수 있다. 무작위 샘플링은 이러한 상관관계를 깨뜨려 일반화 성능을 높이고 학습을 안정시킨다.</b>
    </div>
    </details> <br/>

   
2. 메인 루프의 보상 설계 코드 `shaped_reward = reward if not done or score == 499 else -100.0`에서, 에피소드가 499점에 도달하기 전에 실패했을 때만 `-100`이라는 큰 페널티를 주는 주된 이유는 무엇일까?
   
    A.에피소드의 총점이 음수가 되지 않도록 방지하기 위해  
    B.에이전트가 가능한 한 오래 버티도록 장려하고, 중간에 실패하는 것을 명확한 '나쁜 행동'으로 학습시키기 위해  
    C.신경망의 가중치가 너무 커지는 것을 방지하기 위해  
    D.리플레이 버퍼에 다양한 종류의 보상을 저장하기 위해  

    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <b>B. 에피소드를 조기에 종료시키는 행동의 가치를 크게 낮춤으로써, 에이전트는 해당 상태-행동 쌍을 피하도록 학습한다.</b>  
    </div>
    </details> <br/>  

3. `agent.update_target_model()` 함수는 에피소드가 끝났을 때(`if done:`)만 호출된다. 매 스텝마다 업데이트하는 방식과 비교했을 때, 이러한 '에피소드 단위' 업데이트 방식의 잠재적인 장점은 무엇일까?
   
    A. 한 에피소드 동안 타겟 네트워크를 고정시켜 일관된 학습 목표를 제공함으로써 학습의 안정성을 높일 수 있다.  
    B. 모델의 가중치를 복사하는 데 드는 계산 비용을 줄여 전체 학습 속도를 높일 수 있다.  
    C. 메인 네트워크가 타겟 네트워크를 완전히 앞서나가는 것을 방지한다.  
    D. 에피소드가 성공적으로 끝났을 때의 좋은 경험만 타겟 네트워크에 반영할 수 있다.  

    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <b>A. 매 스텝마다 타겟을 업데이트하면 학습 목표가 너무 자주 바뀌어 학습이 불안정해질 수 있다. 주기적으로 고정된 타겟을 사용하는 것이 DQN의 핵심 아이디어 중 하나이다.</b>
    </div>
    </details> <br/>  

4. DQN은 타겟 네트워크를 사용함에도 불구하고 Q-값을 과대평가(Overestimation)하는 경향이 있다. 이를 해결하기 위해 제안된 Double DQN의 핵심 아이디어는 무엇일까?
    
    A. 두 개의 타겟 네트워크를 동시에 사용하여 그 평균값을 타겟 Q-값으로 사용한다.  
    B. 신경망의 마지막 레이어에 드롭아웃(Dropout)을 추가하여 모델의 일반화 성능을 높인다.  
    C. 리플레이 버퍼에서 샘플링할 때, Q-값이 과대평가된 샘플은 제외한다.  
    D. 다음 상태에서 최적 행동을 '선택'하는 것과, 그 행동의 가치를 '평가'하는 것을 서로 다른 네트워크(메인 네트워크와 타겟 네트워크)로 분리한다.  

    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <b>D. Double DQN은 메인 네트워크를 이용해 다음 상태에서 Q-값이 가장 큰 행동을 고르고, 그 행동의 Q-값은 타겟 네트워크를 통해 평가하여 과대평가 문제를 완화한다.</b>
    </div>
    </details> <br/>  

5. 위 퀴즈의 정답 (D) 내용을 바탕으로, 기존 DQN 코드를 Double DQN으로 수정하기.

    [기존 DQN 코드 (Before)]

    ```python
    # train_model 메서드 내부...

    # 타겟 네트워크로 다음 상태의 Q-값들 예측
    target_val = self.target_model.predict(next_states, verbose=0)

    for i in range(self.batch_size):
        if dones[i]:
            target[i][actions[i]] = "뭘까요?"
        else:
            # 기존 DQN: 타겟 네트워크가 예측한 Q-값 중 가장 큰 값을 그대로 사용
            target[i][actions[i]] = "뭘까요?" + self.discount_factor * "뭘까요?"
    ```

    [수정할 Double DQN 코드 (After)]
    아래 코드의 빈칸 (뭘까요?) 3개를 채워 Double DQN을 완성해 보자.

    ```python
    # train_model 메서드 내부...

    # 1. 행동 '선택'을 위해 메인 네트워크로 다음 상태의 Q-값 예측
    next_q_values_main = self.model.predict(next_states, verbose=0)

    # 2. 가치 '평가'를 위해 타겟 네트워크로 다음 상태의 Q-값 예측
    target_val = self.target_model.predict(next_states, verbose=0)

    for i in range(self.batch_size):
        if dones[i]:
            target[i][actions[i]] = rewards[i]
        else:
            # Step 1: 메인 네트워크의 예측값으로 최적 행동의 '인덱스'를 선택
            action = "뭘까요?"

            # Step 2: 타겟 네트워크의 예측값에서 위에서 선택한 행동의 '가치'를 평가
            value = "뭘까요?"
            
            # Step 3: 위에서 평가한 가치(value)를 이용해 최종 타겟 계산
            target[i][actions[i]] = rewards[i] + self.discount_factor * "뭘까요?"
    ```

    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <code>

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
    </code>
    </div>
    </details> <br/>  

6. Prioritized Experience Replay(PER)에서 경험의 '우선순위'를 결정하는 가장 중요한 기준은 무엇일까?

    A. 해당 경험에서 얻은 보상(Reward)의 크기  
    B. 신경망의 예측과 실제 목표 값 사이의 오차 (TD-error)  
    C. 얼마나 최근에 수집된 경험인지 (Recency)  
    D. 해당 경험의 상태(State)가 얼마나 복잡한지  

    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <b>B. TD-error가 크다는 것은 에이전트의 예측이 많이 빗나갔다는 의미이며, 이는 곧 배울 점이 많은 '놀라운' 경험이라는 뜻이다. PER은 이 TD-error가 큰 샘플을 더 자주 뽑아 효율적으로 학습한다.</b>
    </div>
    </details> <br/>  

7. Dueling DQN의 신경망은 내부적으로 두 개의 스트림(stream)으로 나뉘어 값을 예측한 후 하나로 합친다. 이 두 스트림이 각각 예측하는 것은 무엇일까? (주관식)


    <details>
    <summary><b>정답</b></summary>
    <div markdown="1">
    <b>Dueling DQN은 '이 상태 자체가 얼마나 좋은가?'를 나타내는 상태 가치(V(s))와 '이 상태에서 각 행동이 평균적으로 얼마나 더 나은가?'를 나타내는 어드밴티지(A(s,a))를 분리하여 학습한다.</b>
    </div>
    </details> <br/>  