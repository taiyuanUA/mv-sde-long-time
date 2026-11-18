# 第二个模型：三次漂移与乘性噪声的长时间混沌传播

## 1. 实验对象及执行边界

本次使用一个新模型，未沿用加性噪声实验的数据：

\[
 dX_t=[-2X_t-14X_t^{[3]}+\tfrac12\mathbb EX_t]dt+S(X_t)dW_t,
 \quad S(x)=\operatorname{diag}(\sqrt{1+x_1^2},\ldots,\sqrt{1+x_d^2}),
 \quad X_0\sim N(0,I_d).
\]

- 主实验 T=100，h=1/64，R=64 个**独立完整粒子系统**。
- N=40,80,160,320,640,1280,2560，d=2,3,4,5,8。
- 每次先生成 N 个独立八维 Gaussian 初值，低维情形由坐标前缀得到。系数按坐标分离，所以这是正确的低维模型，而不是按 d/N 缩放误差。
- 同一对相互作用/非相互作用轨道共享初值及 Brownian 增量，但扩散系数分别在两条轨道自己的中间值上计算。
- 相互作用系统每步仍然计算自己的随机经验均值，不能设为零。
- 从 t=0 积分到 T=100，没有用平稳初始化、截短终点模拟或矩方程替代路径计算。
- 原始结果由 `verification/run_experiment.cpp` 实际编译运行。MATLAB 文件实现相同 SSBE 算法；本环境没有 MATLAB/Octave，未在 MATLAB 解释器运行该文件。不同 RNG 会产生正常的统计差异。
- 两个系统都以同一步长离散。因此误差是一个带时间离散偏差的 PoC 均方误差估计，不是精确粒子流误差的无偏估计。

## 2. 假设 1--4 的验证

取 q0=80、kappa=3。令 g(u)=sqrt(1+u²)，r=x-y，Delta m=m(mu)-m(nu)，
B(x,y)=sum_l (x_l-y_l)²(x_l²+x_l y_l+y_l²)。

任意概率耦合给出
\[
 |m(\mu)-m(\nu)|\le W_2(\mu,\nu),\qquad |m(\mu)|\le W_2(\mu,\delta_0).
\]

由平方差公式及分母不小于2，
\[
 |g(u)-g(v)|^2
 =\frac{(u-v)^2(u+v)^2}{(\sqrt{1+u^2}+\sqrt{1+v^2})^2}
 \le\frac13(u-v)^2(u^2+uv+v^2).
\]
这里使用 3(u+v)² <= 4(u²+uv+v²)。另外 |g'(u)|<=1，故 g 全局1-Lipschitz。

### 假设1
\[
\begin{aligned}
2\langle r,b(x,\mu)-b(y,\nu)\rangle+79\|S(x)-S(y)\|^2
&\le-4|r|^2-(28-79/3)B(x,y)+\langle r,\Delta m\rangle\\
&\le-\tfrac72|r|^2+\tfrac12 W_2(\mu,\nu)^2.
\end{aligned}
\]
所以 K1=7/2，K2=1/2。这说明为何系数14足以吸收80阶矩条件的扩散贡献。

### 假设2
\[
 |x^{[3]}-y^{[3]}|\le\tfrac32(|x|^2+|y|^2)|x-y|.
\]
所以
\[
 |b(x,\mu)-b(y,\nu)|^2\le
 2646[(1+|x|^4+|y|^4)|x-y|^2+W_2(\mu,\nu)^2].
\]
可取 L=2646、kappa=3，b(0,delta0)=0。

### 假设3
\[
 \|S(x)\|_{\rm HS}^2=d+|x|^2.
\]
由 Young 不等式及 79z²-28z⁴ <= 79²/112，
\[
 2\langle x,b(x,\mu)\rangle+79\|S(x)\|^2
 \le-\tfrac72|x|^2+\tfrac12W_2(\mu,\delta_0)^2+
 (79+79^2/112)d.
\]
因此 a1=7/2、a2=1/2、C0=135d 合法。Gaussian 初值具有所有有限阶矩。

### 假设4
\[
 \|S(x)-S(y)\|_{\rm HS}^2\le|x-y|^2.
\]
可取 L_sigma=1。以上常数均不依赖于粒子数 N。

## 3. 为什么有 d/(5N) 的一致时间界

漂移的无均值部分是奇函数，S 是偶函数。将初值和 Brownian motion 同时变号，解也变号；由分布唯一性，极限解保持中心对称，所以 E X_t=0。

非相互作用副本满足
\[
 dX_t^i=(-2X_t^i-14(X_t^i)^{[3]})dt+S(X_t^i)dW_t^i.
\]
Itô 公式给出
\[
 \frac d{dt}\mathbb E|X_t^i|^2
 =-3\mathbb E|X_t^i|^2-28\mathbb E\sum_l(X_{t,l}^i)^4+d
 \le-3\mathbb E|X_t^i|^2+d.
\]
因初值二阶矩为 d，有 sup_t E|X_t^i|² <= d。

记 e_i=X_i^N-X_i、bar e=N^{-1}sum e_i、zeta=N^{-1}sum X_i，
u_N=N^{-1}sum E|e_i|²。独立真解副本零均值，故 E|zeta|²<=d/N。

同步耦合在乘性噪声下不会消去噪声项，但差分扩散的平方可用上面的 g 差分界吸收进负三次项。于是
\[
 u_N'\le -4u_N+E|\bar e|^2+E\langle\bar e,\zeta\rangle
 \le-\tfrac52u_N+\tfrac d{2N},\qquad u_N(0)=0.
\]
因此
\[
 \sup_{t\ge0}\sup_i E|X_t^{i,N}-X_t^i|^2\le\frac{d}{5N}.
\]
这只是上界，不是精确误差。平均场只通过均值依赖，是比一般Wasserstein上界更快的原因。无需宣称该模型的理论--数值差距无法解释。

## 4. 两个系统的 SSBE

中间阶段：
\[
 (1+2h)Z_i+14hZ_i^{[3]}-\tfrac h2\overline Z=Y_i,
 \qquad\overline Z=N^{-1}\sum_j Z_j.
\]
更新：
\[
 Y_i^+=Z_i+S(Z_i)\Delta W_i.
\]
非相互作用参考的隐式阶段去掉均值项，更新时使用 S(tilde Z_i)，而不是强制用 S(Z_i)。

每个坐标的 Newton Jacobian 为
\[
 J=\operatorname{diag}(1+2h+42h Z_i^2)-\frac{h}{2N}{\bf1}{\bf1}^T.
\]
用 Sherman--Morrison 公式解每次 Newton 系统，停止准则为归一化非线性残差不超过 1e-12。超过20次更新或遇非有限状态即报错，不用未收敛结果继续模拟。

所用步长是本次数值设置；没有声称 h=1/64 满足论文通用余项常数算出的每个充分步长阈值。

## 5. 误差、回归与统计

对每个独立完整系统 r，先计算
\[
 A_{r,N,d}(t)=\frac1N\sum_{i=1}^N|Y^{i,N,(r)}_{t/h}-\widetilde Y^{i,(r)}_{t/h}|^2.
\]
主图使用
\[
 \widehat E_{N,d}(100)=\frac1R\sum_{r=1}^R A_{r,N,d}(100).
\]
**不取平方根**。按每0.25时间单位保存误差。长期窗口统计先对[80,100]保存点平均平方误差，再对系统平均。使用全部七个N进行 log(error)=c+slope*log(N) 自由回归；只有参考线的斜率预先固定为-1。不同维数共享随机数，不作为五套独立统计证据。

5000次 bootstrap 以完整系统为重采样单位，保留维数间共享随机数的关系。区间只量化 Monte Carlo 不确定性，不包括离散偏差。

## 6. 步长检查

另取N=2560、R=32，在同一初值、同一Brownian路径下，从0到100同时计算h=1/64和h/2=1/128的两组“粒子--极限参考”耦合过程，粗增量严格等于两个细增量之和。

报告两种不同指标：

1. 标量PoC均方误差的相对变化 E_h/E_{h/2}-1；
2. 两个耦合误差过程自身差的RMS，即 ||(Y_h-X_h)-(Y_{h/2}-X_{h/2})||_2。

它们不同，不能混写。两级差并不构成到真连续解误差的严格上界。实际结果见 `data/refinement_summary.csv`。

## 7. 文件及复现

MATLAB不调用Python或C++：

```matlab
addpath('matlab');
multiplicative_poc_T100('plot'); % 读取归档统计表，不重新模拟
multiplicative_poc_T100('test');
multiplicative_poc_T100('simulate','my_results',64);
multiplicative_poc_T100('refine','my_refinement',32);
```

MATLAB代码包含模拟、绘图、统计与步长检查，归档数据和新运行结果可以分别使用。用MATLAB重跑时随机数发生器不同，因此不应期待逐位相同的曲线。

实际执行程序：

```bash
g++ -O3 -march=native -std=c++17 -fopenmp verification/run_experiment.cpp -o run_experiment
OMP_NUM_THREADS=4 ./run_experiment --R 64 --out data
OMP_NUM_THREADS=4 ./run_experiment --R 32 --N 2560 --refine --out data
```

`verification/analyze.py` 和 `verification/make_figures.py` 仅用于独立审计和归档作图；MATLAB复现不需要它们。

本目录不重复附送之前过期的文章或旧实验资料。本次模型与参数均来自当前消息，结果只来自本轮计算。


实际主实验和步长检查的数值汇总见 `RESULTS_ZH.md`。终端斜率与长期窗口斜率均原样报告。
