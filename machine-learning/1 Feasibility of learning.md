# Feasibility of Learning

"학습이 가능한가?"

머신러닝은 주어진 데이터셋에 대해 가설을 세우고, 여러 가설들 중에 가장 성능이 좋은 모델을 뽑아 이상적인 함수를 찾는 과정이다. 이렇게 했을 때 과연 이상적인 함수를 찾을 수 있을지, 즉 주어진 데이터셋이 있다면 이상적인 학습이 가능할지 수학적으로 확인하자.

## Hoeffding inequality

Chebyshev inequality의 확장판

$$
\mathbb{P}[|\hat \mu - \mu| > \epsilon]\leq 2e^{-2\epsilon ^2 N}
$$

확률변수 review

- $\mu$는 전체 데이터의 "실제" 평균
- $\hat\mu$는 $N$개 뽑아서 계산한 모평균
- $N$개 뽑아서 모평균이 실제 평균에 가까울 확률은 bounded 되어 있음

직관적으로 모평균은 실제 평균에 가깝다는 의미를 가진 식이다.

## Multiple hypothesis

머신러닝은 데이터셋들을 가지고 함수 $f: \mathcal{X} \rightarrow \mathcal{Y}$에 근접한 함수 $g$를 찾는 것이다. $g$를 찾기 위해 가설 $h$를 세우고 맞는지 안 맞는지 보고, 조정해가면서 $g$를 찾는다. (train하고 val loss를 계산해서 최종 모델을 결정하는 방법 생각하면 됨)

가설 $h$에 대한 **in-sample error**를 정의하자.

$$
E_{in}(h) = \frac{1}{N} \sum^N_{n=1} \llbracket h(x_n) \neq f(x_n) \rrbracket
$$

$\llbracket \ \rrbracket$ 브라켓은 각 데이터 $x_i$에 대한 결과값이 다를 경우 1, 같을 경우 0을 의미한다. 따라서 $E_{in}$는 "데이터셋 $N$개를 뽑았을 때 가설 $h$에 대해 $f$와 다른 정도를 측정한 것" 이다. 예를 들어 데이터 10개를 학습시켰을 때 classification을 하나만 틀렸다면 $E_{in}$ 값은 $0.1$이 된다.

이번에는 **out-sample error**를 정의하자. 

$$
E_{out}(h) = \mathbb{P}[h(x) \neq f(x)]
$$

이번에는 데이터셋 $N$개가 아니라 $h$와 $f$의 결과값이 다를 확률이다. 즉 이 경우인 out-sample error가 "실제 확률", in-sample error가 N개 뽑아서 측정한 모평균에 해당한다. 

$$
E_{in} \rightarrow \hat \mu
\ , \ 
E_{out} \rightarrow \mu
$$

앞의 Hoeffding inequality를 이용해서 다음과 같이 쓸 수 있다.

$$
\mathbb{P}[|E_{in}(h) - E_{out}(h)| > \epsilon]\leq 2e^{-2\epsilon ^2 N}
$$

머신러닝은 가설 $h$들 중 가장 성능이 좋은 $g$를 찾는 것이라 했다. Hypothesis set $\mathcal{H}$에 $M$개의 $h$가 있다고 할 때, $g$에 대한 error를 생각하면 다음과 같이 확장할 수 있다.

$$
\mathbb{P}[|E_{in}(g) - E_{out}(g)| > \epsilon]\leq 2Me^{-2\epsilon ^2 N}
$$

의미를 해석하면 목표하는 함수 $g$가 맞을 "N개 데이터 측정 확률"가 실제 $g$의 정확도에 가까울 확률은 bound 되어 있다는 것이다. 여기서 가설의 개수 $M$은 유한해야 한다. 

## Feasibility of learning

앞에서 정의한 함수들을 이용해 가장 처음의 질문 "학습이 가능한가?"에 답해보자. 학습이 가능하다는 것은 $g$의 오차가 0에 가깝다는 것이므로 다음 두 가지 조건으로 분리할 수 있다.

- N개 데이터를 뽑아서 확인한 오류 비율이 실제 오차 확률과 비슷한가?
- N개 데이터의 오류 비율이 0에 근접한가?

다시 말하면,

- $E_{in}(g) \approx E_{out}(g)$ 인가?
- $E_{in}(g) \approx 0$ 인가?

만약 이 둘을 만족한다면 $E_{out}(g) \approx 0$이 되어 "실제 오차 확률은 0에 가깝다", 즉 학습이 되었다고 할 수 있다.

앞의 M개 가설에 대한 Hoeffding inequaility에서 $E_{in}(g)$가 $E_{out}(g)$에 충분히 가까움을 확인했다. 1번 조건은 해결되었다. 2번 조건 $E_{in}(g) \approx 0$는 실제 머신러닝 방법론에서 해결한다. 

따라서 "학습이 가능한가?"는 "오차를 매우 작게 만들 수 있는가?"로 함축할 수 있다. 이제 머신러닝으로 오차를 줄이는, optimize하는 방법을 공부하면 된다.