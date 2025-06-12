---
title: SLAM 中的基础数学知识
date: 2024-03-07 15:04:23
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: complete
destination: 03-Projects/01-Note/01-SLAM
share: false
obsidianUIMode: source
number headings: auto, first-level 1, max 6, _.1.1
---



# 1 旋转的表示
## 1 旋转向量

## 2 旋转矩阵
1. 旋转矩阵有且只有一个实数特征值1；
2. 不考虑单位矩阵，旋转矩阵的实数特征值对应的特征向量即是axis-angle表达式的旋转轴，旋转矩阵的迹与旋转角度相关。
## 3 四元数
1. [球面线性插值（Spherical linear interpolation，slerp)](https://cnblogs.com/21207-iHome/p/6952004.html)
![[1-SLAM 中的基础数学知识.png]]
$$
Slerp(p,q,t) = \frac{\sin[(1-t)\theta] + \sin(t \theta q)}{\sin(\theta)}
$$
## 4 旋转表示之间的相互转换
1. 四元数转旋转向量 
$$
\begin{aligned}
\left\{
\begin{array}{l}
\theta = 2\arccos s \\\\
[n_x,n_y,n_z]^T = \frac{v^T}{\sin{\frac{\theta}{2}}}
\end{array}
\right.
\end{aligned}
\tag{1}
$$
2. 旋转$\theta\boldsymbol{n}$向量转四元数
$$
\boldsymbol{q} = [\cos \frac{\theta}{2},\boldsymbol{n} \sin \frac{\theta}{2}] \tag{2}
$$
3. 旋转矩阵转旋转向量
$$
\begin{cases}
\theta = \arccos(\frac{tr(\boldsymbol{R})-1}{2}) \\\\[1ex]
n = Rn, \quad \text{轴n是R特征值为1的单位特征向量}
\end{cases}
\tag{3}
$$
4. 旋转向量$\boldsymbol\omega$转旋转矩阵（*罗德里格斯公式*）
$$
\boldsymbol{R} = \cos\theta \boldsymbol{I} + (1- \cos\theta ) \boldsymbol {n} \boldsymbol {n}^T + \sin\theta  \boldsymbol {n}^{\wedge} = exp{(\boldsymbol{\omega^{\wedge}})}
\tag{4}
$$
*旋转向量转旋转矩阵本质就是对应李代数的指数映射，李代数的指数映射将旋转向量映射为反对称矩阵*
5. 旋转矩阵转四元数
$$
\begin{cases}
\boldsymbol\omega = \frac{\sqrt{1 + \text{trace}(\boldsymbol{R})}}{2} \\\\[1ex]
\boldsymbol{x} = \frac{R_{32} - R_{23}}{4w} \\\\[1ex]
\boldsymbol{y} = \frac{R_{13} - R_{31}}{4w} \\\\[1ex]
\boldsymbol{z} = \frac{R_{21} - R_{12}}{4w}
\end{cases}
\tag{5}
$$
$$
q = sqrt((R_{11} + R_{22} + R_{33} + 1) / 4) * [1, R_{23} - R_{32}, R_{31} - R_{13}, R_{12} - R_{21}]
\tag{6}
$$
6. 四元数转旋转矩阵
$$
\boldsymbol{R} = \boldsymbol{v} \boldsymbol{v}^T + s^2 + \boldsymbol{I} + 2s \boldsymbol{v}^{\wedge} + (\boldsymbol{v^{\vee}})^2
\tag{7}
$$

7. 二维旋转角转旋转矩阵 
$$
\theta = 
\begin{bmatrix} 
\cos(x) & -sin(x) \\\\
sin(x) & cos(x)
\end{bmatrix} 
\tag{8}
$$
8. 欧拉角(roll,pitch,yaw)转旋转矩阵
$$
(roll,pitch,yaw) = (\alpha,\beta,\phi) = 
\begin{cases}
	R_{roll} &=& 
		\begin{bmatrix}
		1 & 0 & 0 \\\\
		0 & cos(\alpha) & -sin(\alpha) \\\\
		0 & sin(\alpha) & cos(\alpha)  \\\\
		\end{bmatrix} \\\\[1ex]
	R_{pitch} &=&
		\begin{bmatrix}
		cos(\beta) & 0 & sin(\beta) \\\\
		0 & 1 & 0 \\\\
		-sin(\beta) & 0 & cos(\beta) \\\\
		\end{bmatrix} \\\\[1ex]
	R_{yaw} &=&
		\begin{bmatrix}
			cos(\phi) & -sin(\phi) & 0 \\\\
			sin(\phi) & cos(\phi) & 0 \\\\
			0 & 0 & 1 \\\\
		\end{bmatrix} \\\\[1ex]
	
	R_{(\alpha,\beta,\phi)} &=& R_{\phi}*R_{\beta}*R_{\alpha}, \text{一般采ZYX的顺序进行旋转叠加}
\end{cases}
\tag{9}
$$
9. 计算向量与坐标轴之间的 rpy角
$$
\begin{cases}
	roll &=& \tan(z/y) \\\\
	pich &=& \tan(x/z) \\\\
	yaw  &=& \tan(y/x) \\\\
\end{cases}
$$

10. 四元数转欧拉角（RPY）
![[2-SLAM 中的基础数学知识.png]]
## 5 区别
1. 四元数表示的旋转，可以更方便的进行插值运算

*纯虚四元数的指数映射结果为单位四元数，而单位四元数可以表示空间中的旋转，所以四元数的旋转可以由纯虚四元数的指数映射表示*
$$
\boldsymbol{q} = exp(\boldsymbol{\tilde{w}}), \quad \boldsymbol{\tilde{w}}\text{表示纯虚四元数}
$$
**四元数表示旋转时，旋转的角度只有李代数和旋转向量表示的一半，因为使用四元数进行旋转变换时需要进行两次乘法计算** 
$$\boldsymbol{p^{'}} = \boldsymbol{q} \boldsymbol{p} \boldsymbol{p^{-1}}$$
其中$\boldsymbol{p^{'}} 和 $\boldsymbol{p}$ 分别为纯虚四元数表示的3d点坐标。

2. 李代数更适合表示连续的空间变化，其具有较好的微分性质，在导航和控制问题中更容易处理
3. 一般传感器的测量结果都是角度（例如：IMU），角度可以很方便的转换为四元素，因此其计算非常高效，因此在 IMU 等高频计算场景中一般旋转四元素。但是在视觉 SLAM 中一般用旋转矩阵表示旋转，旋转矩阵可以很方便的通过指数映射转为李代数，并且视觉 SLAM 中需要直接对旋转进行优化，而李代数求导十分方便，所以更适合用李代数。（*总结：角度的增量和插值计算用四元数，涉及旋转的导数运算用李代数*）
## 6 为什么最小乘的解为最小特征值对应的特征向量
求一个最小二乘问题 
$$
	min||Ax = b||_2^2, \quad x \neq 0 \quad and \quad |x| = 1
$$
首先我们要理解的是求 $Ax=b$ 的解就是求向量b在A矩阵的列空间（以列向量为基地张成的空间）中的像。假设$A=I$那么A的列空间就是欧式空间。如果$A \neq I$ ,例如：
$$
A = 
\begin{bmatrix}
 2 & 0 & 0 \\\\
 0 & 1 & 0 \\\\
 0 & 0 & 1
\end{bmatrix}
$$
那么b在A的列空间中的像的x轴就会被压缩为1/2。这也就是我们常说的特征值表示的是列空间在各个列向量方向上的缩放比例。
接下来我们再看$|x|=1$,很明显在欧式空间中表示为$r=1$的球面构成的集合。但是由于b在A的列空间中的像的各轴坐标会被缩放，所以$|x|=1$在A的列空间中形成的是一个椭球。而求解 $min||Ax = b||_2^2$ 的问题即为在椭球面上找一点，使其到原心的距离最短。根据椭圆的知识我们很容易知道，椭圆上离原心最近的点为短轴与椭圆的交点。短轴则对应的是最小特征向量对应的坐标轴。也即最小二乘的解为最小特征值对应的特征向量。
反之 $max||Ax = b||_2^2$  的解为最大特征值对应的特征向量


## 7 为什么信息矩阵是协方差的逆
协方差表示了不同特征之间的相关性。而根据逆矩阵的伴随矩阵求法可知，的逆矩阵的第 i，j 个元$A_{i,j}$为去掉原矩阵的第i行和第j列之后的剩余元素的行列式与A矩阵行列式的比值。
$$
A^{-1} =\frac{A^*}{|A|}, \quad  A^*_{i,j} 为去掉A_{i,j}所在的行和列之后剩余元素的行列式
$$



## 8 KF、EKF、ESKF
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
![[1-SLAM 基础.png]]

# 10 常用数学公式 
## 1 利群与李代数转换
$$
\begin{aligned}
exp(\phi^{\wedge}) &=R \\\\[1ex]
\phi &= log(R)^{\vee} \\\\[1ex]
\phi^{\wedge} &= log(R)
\end{aligned}
$$


### 1.1 叉积交换
$$
\begin{aligned}
	a \times b = - b \times a \\\\
	\partial\theta^{\wedge} \omega = \omega^{\wedge} \partial\theta
\end{aligned}
$$
### 1.2 SO(3) 伴随公式 
$$
\phi^{\wedge}R = R(R^T\phi)^{\wedge}
$$
### 1.3 指数函数常用性质 12445
$$
\begin{cases}
e^{x} = 1 + x + \cdots + \dfrac{x^n}{n!} \\\\[8pt]
e^{-ax} = e^{-a} + e^{x} \\\\[8pt]
e^{a+b} = e^{a}e^{b}
\end{cases}
\quad\Rightarrow\quad
\begin{cases}
\exp(x) = 1 + x \\\\[8pt]
\exp(\theta + \Delta\theta) = \exp(\theta)\exp(\Delta\theta)
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

## 8  SO(3)上的BCH近似公式
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


### 1.4 旋转求导
$$
\begin{cases}
\dot{R} = Rw^{\wedge}, \quad \text{该式也称为泊松方程，w为瞬时角速度}

\\\\[1ex]

\dot{exp(\delta\theta^{\wedge})} = exp(\delta\theta^{\wedge})exp(\delta\dot{\theta}^{\wedge})
\end{cases}
$$

# 几何变换
## 1 平面系数与平面法向量 
平面方程
$$
ax+by+cz+d = 0
$$
平面与各轴的交点：
$$
\begin{cases}
P_x : (-\frac{d}{a},0,0) \\\\
P_y : (0,-\frac{d}{b},0) \\\\
P_z : (0,0,-\frac{d}{c})
\end{cases}
$$
点P(x,y,z)到平面$\vec{n}$的距离 
$$
	d_p = \frac{|\vec{PP_x} \vec{n}|}{|\vec{n}|}
$$



# Appendix
```c++
geometry_msgs::Quaternion EulerAngletoQuaternion(double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
  // Abbreviations for the various angular functions
  double cy = cos(yaw * 0.5);
  double sy = sin(yaw * 0.5);
  double cp = cos(pitch * 0.5);
  double sp = sin(pitch * 0.5);
  double cr = cos(roll * 0.5);
  double sr = sin(roll * 0.5);

  geometry_msgs::Quaternion q;
  q.w = cy * cp * cr + sy * sp * sr;
  q.x = cy * cp * sr - sy * sp * cr;
  q.y = sy * cp * sr + cy * sp * cr;
  q.z = sy * cp * cr - cy * sp * sr;
  return q;
}

void QuaterniontoEulerAngle(const geometry_msgs::Quaternion &q, double &roll,
                            double &pitch, double &yaw) {
  // roll (x-axis rotation)
  double sinr_cosp = +2.0 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = +1.0 - 2.0 * (q.x * q.x + q.y * q.y);
  roll = atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = +2.0 * (q.w * q.y - q.z * q.x);
  if (fabs(sinp) >= 1)
    pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
    pitch = asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = +2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  yaw = atan2(siny_cosp, cosy_cosp);
}

double angle_normal(double original_ang) {
  if (original_ang > 3.1415926)
    original_ang -= piX2;
  else if (original_ang < -3.1415926)
    original_ang += piX2;

  return original_ang;
}

double angles_weight_mean(double angleA, double weightA, double angleB, double weightB) { //弧度
//  double angleA_cal;
//  double angleB_cal;
  double mean{}, diff{};

  if (angleA * angleB < 0) {
    if (angleA < 0) {
      diff = pi + angleA + pi - angleB;
      if (diff < pi)
        mean = angleB * weightB + (angleB + diff) * weightA;
      else
        mean = angleA * weightA + angleB * weightB;
    }

    if (angleB < 0) {
      diff = pi + angleB + pi - angleA;
      if (diff < pi)
        mean = angleA * weightA + (angleA + diff) * weightB;
      else
        mean = angleA * weightA + angleB * weightB;
    }
  } else {
    mean = angleA * weightA + angleB * weightB;
  }

  return angle_normal(mean);
}
```



平面对齐：
```cpp
    // 1. 计算两平面的旋转
    // 法线叉乘计算两平面的旋转轴
    Eigen::Vector3d rot_axis2 = slave_gplane.normal.cross(master_gplane.normal);
    rot_axis2.normalize();
    // 法线点乘计算面的旋转角
    double alpha2 = std::acos(slave_gplane.normal.dot(master_gplane.normal));
    Eigen::Matrix3d R_ms;
    // 平面对齐的旋转向量
    R_ms = Eigen::AngleAxisd(alpha2, rot_axis2);
    
    // 2. 计算两平面的平移
    // 平面于z轴的交点 intercept-截距  normal-法线
	Eigen::Vector3d slave_intcpt_local( 0, 0, -slave_gplane.intercept / slave_gplane.normal(2));
    // 交点旋转到对准点云
    Eigen::Vector3d slave_intcpt_master = R_ms * slave_intcpt_local;
    // 计算平面对齐的平移变量 ？？ 为什么只移动z轴坐标，这样可以将平面重合，平面上的点也能对应吗？
    Eigen::Vector3d t_ms(0, 0, t_mp(2) - slave_intcpt_master(2));
```


