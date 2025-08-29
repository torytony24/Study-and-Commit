# Linear Classification

상황설명


## Review

- $E_{in}(h) = \frac{1}{N} \sum^N_{n=1} \llbracket h(x_n) \neq f(x_n) \rrbracket$
- $E_{out}(h) = \mathbb{P}[h(x) \neq f(x)]$

$E_{in}$은 N개 뽑아서 오차 계산한 모평균, $E_{out}$은 실제 오차 확률

## Generalization error

tolerance level $\delta$를 정의하자. "이 정도 범위까지 bound는 용인하자"의 의미를 가지고 있다. 

$$
E_{out}(g) \leq E_{in}(g) + \sqrt{\frac{1}{2N} \ln \frac{2M}{\delta}}
$$

이 식의 루트와 ln은 사실 Hoeffding equation에서 온 거다.

$|E_{out} - E_{in}| \leq \epsilon$ 에서 $E_{out} \leq E_{in} + \epsilon$ 라 한다면 Hoeffding equation에 의해 $\delta = 2Me^{-2\epsilon^2N}$ 라 할 수 있다. 이를 반대로 $\epsilon$에 대해 풀면 위의 식이 나온다. 

tolerance level $\delta = 0.05$라 한다면 N개 뽑아서 측정한 오차와 실제 오차가 다를 확률의 범위가 5\%라는 말이다. 직관적으로 식을 이해하자면 실제 오차는 N개 측정한 오차 $+\epsilon$ 만큼 bound 되어있다.

## VC dimension

먼저 hypothesis set $\mathcal{H}$에 대해 dichotomies(이분)를 정의하자. 