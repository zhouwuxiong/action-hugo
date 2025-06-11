---
title: SLAM 基础
date: 2024-03-05 11:29:23
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: complete
destination: 03-Projects/01-Note/01-SLAM
share: true
obsidianUIMode: source
---
## 1 常见符号表示 


## 2  SO(3)上的BCH近似公式
BCH公式给出了李代数上的小量加法与李群上小量乘法之间的关系（**李代数加法 ⇔ 李群乘法**），其线性近似公式广泛应用于各种函数的线性化。
在SO(3)中，某个旋转 $R$ 对应的李代数为 $\phi$，左乘一个微小旋转，记作 $\Delta{R}$，对应的李代数为 $\Delta\phi$，那么在李群上得到的结果就是 $\Delta RR$，而在李代数上，根据BCH近似，为 $J_l({\phi})^{-1}\Delta{\phi} +{\phi}$ 合并后可以简单写成：
$$\Delta{R}{R} = \exp(\Delta{\phi}^{\wedge})\exp({\phi}^{\wedge}) = \exp \left(({\phi} + {J}_l({\phi})^{-1}\Delta{\phi})^{\wedge} \right)$$
反过来，如果在李代数上进行加法，让一个 ${\phi}$加上小量 $\Delta{\phi}$，那么可以近似为李群上带左右雅克比矩阵的乘法：
$$\exp(({\phi} + \Delta{\phi})^{\wedge}) = \exp(({J}_l({\phi})\Delta{\phi})^\wedge) \exp({\phi}^\wedge)= \exp({\phi}^\wedge) \exp(({J}_r({\phi})\Delta{\phi})^\wedge)
$$
其中SO(3)的左雅克比矩阵为
$$
\begin{aligned}
{J}_l(\theta{a}) &= \frac{\sin\theta}{\theta}{I} + (1-\frac{\sin\theta}{\theta}){a}{a}^T+(\frac{1-\cos\theta}{\theta}){a}^{\wedge} \\{J}^{-1}_l(\theta{a}) &= \frac{\theta}{2}\cot\frac{\theta}{2}{I} + (1-\frac{\theta}{2}\cot\frac{\theta}{2}){a}{a}^T-\frac{\theta}{2}{a}^{\wedge}
\end{aligned}
$$
而SO(3)的右雅克比矩阵为
$$
{J}_r({\phi}) = {J}_l(-{\phi})
$$

## 3 KF、EKF、ESKF
**KF 的状态和观测方程的递推是高斯分布的线性变换，融合结果则是两个高斯分布相乘得到一个新的高斯分布，卡尔曼增益则是系数**
[Site Unreachable](https://zhuanlan.zhihu.com/p/39912633)

KF、EKF、ESKF 的本质都是对高斯分布的线性变换。他们的区别在于高斯线性变换时的系数不同，KF 的线性系数是常数，EKF的线性系数是雅可比矩阵
**KF**
假设运动方程为:
$$
x_{k+1} = Ax_k + u_k + w_k, w \sim (\mu,\sigma^2)
$$
由于该公式为线性变换，所以$x_{K+1}$ 也服从高斯分布。
$$
x_{k+1} \sim (A\mu + u_k ,A \sigma^2 A^T)
$$
**EKF**
假设运动方程为:
$$
x_{k+1} = f(x_k) + u_k + w_k, w \sim (\mu,\sigma^2)
$$
其中$f(x)$ 表示非线性变换，将其在$x_k$进行一阶泰勒展开：
$$
f(x) = f(x_k) + J(x-x_k)
$$
所以下一时刻的$x_{k+1}$ 的分布为
$$
x_{k+1} \sim (J \mu + u_k, J \sigma^2 J^T)
$$
其与KF 的区别在与 A 是常量，而 J 是与线性化点相关的变量。
**ESKF**
ESKF 与 EKF 类似，只是 EKF 是对整个运动方程进行高斯过程，而 ESKF 只对噪声进行高斯过程，对于测量值（名义变量）则进行直接的递推。最终的结果由测量值的递推和噪声的预测值相加得到。

**高斯分布的线性变换**
若$x\sim(\mu,\sigma^2)$，则$Ax+b \sim (A\mu+b,A\sigma^2A^T)$
**高斯分布相乘**
若$x\sim(\mu_0,\sigma_0^2)$，$y\sim(\mu_1,\sigma_1^2)$，则$x*y \sim (A\mu+b,A\sigma^2A^T)$
![1-SLAM 基础.png](1-SLAM%20%E5%9F%BA%E7%A1%80.png)

## 4 常用数学公式 
## 5 利群与李代数转换
$$
\begin{aligned}
exp(\phi^{\wedge}) &=R \\\\[1ex]
\phi &= log(R)^{\vee} \\\\[1ex]
\phi^{\wedge} &= log(R)
\end{aligned}
$$


### 5.1 叉积交换
$$
\begin{aligned}
	a \times b = - b \times a \\\\
	\partial\theta^{\wedge} \omega = \omega^{\wedge} \partial\theta
\end{aligned}
$$
### 5.2 SO(3) 伴随公式 
$$
\phi^{\wedge}R = R(R^T\phi)^{\wedge}
$$
### 5.3 指数函数常用性质 1245657
$$
\begin{cases}
e^{x} = 1 + x +  + \frac{x^n}{n!} 
e^{-ax} = e^{-a} + e^x 
e^{a+b} = e^ae^b
\end{cases}

\quad\Rightarrow\quad

\begin{cases}
exp(x) = 1 + x \\\\[1ex]
exp(\theta + \Delta\theta) = exp(\theta)exp(\Delta\theta)
\end{cases}
$$### BCH 近似
BCH 公式关联了李群乘法与李代数加法之间的关系，**李群（SO(3)）上乘小量等于李代数（so(3)）上加小量（带雅可比矩阵）**
$$
\begin{aligned}
exp(\phi^{\wedge})exp(\Delta\phi^{\wedge}) \approx exp((\phi+J_r^{-1}\Delta\phi)^{\wedge})

\\\\[1ex]

exp(\phi+\Delta\phi) \approx exp(\phi^{\wedge})exp(J_r\Delta\phi^{\wedge}) \approx exp(J_l\Delta\phi^\wedge)exp(\phi^{\wedge})

\\\\[1ex]

Log(R exp(\Delta \phi)) \approx Log(R)+ J_r^{-1}(Log(R))\Delta\phi^{\wedge}

\\\\[1ex]

log(\prod_{k=1}^{j-1}exp(\Delta\phi ))^{\vee} \approx \sum_{k=i}^{j-1}\Delta\phi, \quad \text{由于 $\Delta\phi$ 为小量,所以假定了$J_r=I$}
\end{aligned}
$$

### 旋转求导
$$
\begin{cases}
\dot{R} = Rw^{\wedge}, \quad \text{该式也称为泊松方程，w为瞬时角速度}

\\\\[1ex]

\dot{exp(\delta\theta^{\wedge})} = exp(\delta\theta^{\wedge})exp(\delta\dot{\theta}^{\wedge})
\end{cases}
$$