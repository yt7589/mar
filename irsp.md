# 基于MAR的智能雷达信号处理

# 1. 论文解读
## 1.1. 隐空间定义
$t$时刻雷达接收到的回波信号为：
$$
\boldsymbol{x}^{(t)} \in R^{A}
$$
共有$A$个接收天线。一个Chirp采样$N=256$次，我们设置一个Chirp为一个序列：
$$
X^{(i)} \in R^{N \times A}
$$
我们的任务是根据$\boldsymbol{x}^{(0)}, \boldsymbol{x}^{(2)}, ..., \boldsymbol{x}^{(t-1)}$序列，预测$\boldsymbol{x}^{(t)}$。第$t$时刻帧出现的概率为：
$$
p(\boldsymbol{x}^{(0)}, \boldsymbol{x}^{(2)}, ..., \boldsymbol{x}^{(t)})=\prod_{i=0}^{t}p(\boldsymbol{x}^{(i)} \vert \boldsymbol{x}^{(0)}, \boldsymbol{x}^{(2)}, ..., \boldsymbol{x}^{(i-1)})
$$
我们采用最大似然算法，任务就是使这种情况出现的概率最大。
我们首先通过Transformer网络，求出当前时刻的隐空间向量：
$$
\boldsymbol{z}^{(i)} = f_{T}(\boldsymbol{x}^{(0)}, \boldsymbol{x}^{(2)}, ..., \boldsymbol{x}^{(i-1)}), \quad \boldsymbol{z} \in R^{D}
$$
我们接下来要从隐向量预测下一帧：
$$
p(\boldsymbol{x}^{(i)} \vert \boldsymbol{z}^{(i)})
$$
即在给定$\boldsymbol{z}^{(i)}$的条件下使$\boldsymbol{x}^{(i)}$出现的概率最大。

## 1.2. 去噪扩散模型
系统初始状态为：
$$
\boldsymbol{x}_{(0)} \sim q(\boldsymbol{x}_{(0)}) \tag{1.2.001}
$$
扩散过程或正向过程是向$\boldsymbol{x}_{(0)}$中不断添加噪声的过程，直到$T$步为止：
$$
q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(t-1)}) := \mathcal{N}(\boldsymbol{x}_{(t)}; \sqrt{1-\beta _{t}}\boldsymbol{x}_{(t-1)}, \beta _{t}\mathbb{I}), \quad t = 0, 1, 2, ..., T-1, T \tag{1.2.002}
$$
其中$\beta_{t}$可以视为超参数，虽然可以通过重参数化确定其精确值，但是在实践中，为了简化问题，我们将其视为一个足够小的常数。
我们的模型要使$\boldsymbol{x}_{(0)}$出现的概率最大，因此定义损失函数为：
$$
\mathcal{L}_{f} = \mathbb{E}\left[ -\log{p_{\theta}(\boldsymbol{x}_{(0)})} \right] \tag{1.2.003}
$$
我们可以将上式视为分母为1的式子，我们以概率替换$q(\boldsymbol{x}_{(1:T)}\vert \boldsymbol{x}_{(0)})$，由于概率值小于1，故有：
$$
\mathcal{L}_{f}=\mathbb{E}\left[ -\log{p_{\theta}(\boldsymbol{x}_{(0)})} \right]\le \mathbb{E}_{q}\left[ -\log{\frac{p_{\theta}(\boldsymbol{x}_{(0:T)})}{q(\boldsymbol{x}_{(1:T)}\vert \boldsymbol{x}_{(0)})}} \right] \tag{1.2.004}
$$
下面我们来研究$p_{\theta}(\boldsymbol{x}_{(0)})$，其代表在我们以$\theta$为参数的模型下，通过不断向$\boldsymbol{x}_{(T)}$中去掉噪声，从而出现序列$\{ \boldsymbol{x}_{(T)}, \boldsymbol{x}_{(T-1)}, ..., \boldsymbol{x}_{(0)} \}$的概率，可以表示为：
$$
p_{\theta}(\boldsymbol{x}_{(0:T)}) = p(\boldsymbol{x}_{(T)})\prod_{t=T}^{1}p_{\theta}(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(t)}) \tag{1.2.005}
$$
其中：
$$
p_{\theta}(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(t)}) = \mathcal{N}(\boldsymbol{x}_{(t-1)};\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{(t)},t), \Sigma_{\theta}(\boldsymbol{x}_{(t)}, t)) \tag{1.2.005.1}
$$
将式(1.2.005)代入式(1.2.004)中可得：
$$
\mathcal{L}_{f} = \mathbb{E}\left[ -\log{p_{\theta}(\boldsymbol{x}_{(0)})} \right] \le \mathbb{E}_{q}\left[ -\log{p(\boldsymbol{x}_{(T)})} - \sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(t)})}{q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(t-1)})} } \right] \tag{1.2.006}
$$
我们定义另一个参数：
$$
\begin{cases}
  \alpha_{t} = 1 - \beta_{t} \\
  \bar{\alpha_{t}}=\prod_{s=1}^{t}\alpha_{s}
\end{cases} \tag{1.2.007}
$$
这时正向添加噪声的过程可以表示为：
$$
q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(0)}) = \mathcal{N}(\boldsymbol{x}_{(t)}; \sqrt{\bar{\alpha_{t}}}\boldsymbol{x}_{(0)}, (1-\bar{\alpha_{t}})\mathbb{I}) \tag{1.2.008}
$$
下面我们来化简损失函数，将式(1.2.006)中t=1项从累加号中移出：
$$
\mathcal{L}_{f} = \mathbb{E}_{q}\left[ -\log{p(\boldsymbol{x}_{(T)})} - \sum_{t=2}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(t)})}{q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(t-1)})} } - \log{ \frac{p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)})}{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})} }\right] \tag{1.2.009}
$$
我们定义事件1为在$\boldsymbol{x}_{(0)}$出现下$\boldsymbol{x}_{(t-1)}$且在$\boldsymbol{x}_{(t-1)}$出现下$\boldsymbol{x}_{(t)}$出现的概率：
$$
q(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(0)})q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(t-1)}) \tag{1.2.010}
$$
我们定义事件2为在$\boldsymbol{x}_{(0)}$出现下$\boldsymbol{x}_{(t)}$且在$\boldsymbol{x}_{(t)}$和$\boldsymbol{x}_{(0)}$出现下$\boldsymbol{x}_{(t-1)}$出现的概率：
$$
q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(0)})q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)}) \tag{1.2.011}
$$
我们知道事件1和事件2是同一事件，因此其概率相等，所以有：
$$
q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(t-1)}) = \frac{ q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})q(\boldsymbol{x}_{(t)}\vert \boldsymbol{x}_{(0)}) }{q(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(0)})} \tag{1.2.012}
$$
将式(1.2.012)代入式(1.2.009)可得：
$$
\mathcal{L}_{f} = \mathbb{E}_{q} \Bigg( -\log{p(\boldsymbol{x}_{(T)})} \\
-\sum_{t=2}^{T} \log{ \left( \frac{p_{\theta}(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(t)})}       {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} \cdot \frac{q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(0)})}   \right) } \\
-\log{ \frac{p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)})}{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})} } \Bigg) \tag{1.2.013}
$$
对右式第二项累加进行展开：
$$
\log{\left( \frac{p_{\theta}(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(2)})}{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(2)}, \boldsymbol{x}_{(0)})} \cdot \frac{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(0)})} \right)} + \\
\log{\left( \frac{p_{\theta}(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(3)})}{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(3)}, \boldsymbol{x}_{(0)})} \cdot \frac{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(3)} \vert \boldsymbol{x}_{(0)})} \right)} + \\
... + \\
\log{\left( \frac{p_{\theta}(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(T)})}{q(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(T)}, \boldsymbol{x}_{(0)})} \cdot \frac{q(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} \right)} \tag{1.2.014}
$$
将其进行拆分：
$$
\log{ \frac{p_{\theta}(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(2)})}{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(2)}, \boldsymbol{x}_{(0)})}  + \log{\frac{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(0)})}} } + \\
\log{ \frac{p_{\theta}(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(3)})}{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(3)}, \boldsymbol{x}_{(0)})} + \log{\frac{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(3)} \vert \boldsymbol{x}_{(0)})}} } + \\
... + \\
\log{\frac{p_{\theta}(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(T)})}{q(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(T)}, \boldsymbol{x}_{(0)})} + \log{\frac{q(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})}} } \tag{1.2.015}
$$
将第1列写为累加形式，第2列整理到$\log$中：
$$
\sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})} {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} } + \\
\log{ \bigg( \frac{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(0)})} \cdot \frac{q(\boldsymbol{x}_{(2)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(3)} \vert \boldsymbol{x}_{(0)})} ... \frac{q(\boldsymbol{x}_{(T-2)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(0)})} \cdot \frac{q(\boldsymbol{x}_{(T-1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} \bigg) } \tag{1.2.016}
$$
注意到第2行正好是一个列项公式，所以上式可以化简为：
$$
\sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})} {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} } + \log{ \frac{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} } \tag{1.2.017}
$$
将式(1.2.017)代入式(1.2.013)的第2项得：
$$
\mathcal{L}_{f} = \mathbb{E}_{q} \Bigg( -\log{p(\boldsymbol{x}_{(T)})} \\
-\bigg( \sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})} {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} } + \log{ \frac{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} } \bigg) \\
-\log{ \frac{p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)})}{q(\boldsymbol{x}_{(1)} \vert \boldsymbol{x}_{(0)})} } \Bigg) \tag{1.2.018}
$$
将第2行最后1项与第3行乘到$\log$中得：
$$
\mathcal{L}_{f} = \mathbb{E}_{q} \Bigg( -\log{p(\boldsymbol{x}_{(T)})} \\
 -\sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})} {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} }   \\
-\log{ \frac{p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} } \Bigg) \tag{1.2.019}
$$
将最后一项分母移给第1项：
$$
\mathcal{L}_{f} = \mathbb{E}_{q} \Bigg( -\log{ \frac{p(\boldsymbol{x}_{(T)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} }
 -\sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})} {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} }   
-\log{ p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)}) } \Bigg) \tag{1.2.020}
$$
根据KL散度定义，上式可以写为：
$$
\mathcal{L}_{f} = \mathbb{E}_{q} \Bigg( -\log{ \frac{p(\boldsymbol{x}_{(T)})}{q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)})} }
 -\sum_{t=1}^{T} \log{ \frac{p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})} {q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})} }   
-\log{ p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)}) } \Bigg) \tag{1.2.020}
$$
根据KL散度定义，上式可以写为：
$$
\mathcal{L}_{f} = \mathbb{E}_{q} \Bigg( \\
D_{KL}(q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)}) \Vert p(\boldsymbol{x}_{(T)})) + \\
\sum_{t=2}^{T} D_{KL}(q( \boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)}) \Vert p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})) - \\
\log{ p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)}) } \Bigg) \tag{1.2.021}
$$
我们将上述损失函数分别定义，前向过程损失函数定义为：
$$
\mathbb{L}_{T} = D_{KL}(q(\boldsymbol{x}_{(T)} \vert \boldsymbol{x}_{(0)}) \Vert p(\boldsymbol{x}_{(T)}))  \tag{1.2.022}
$$
反向过程损失函数：
$$
\mathbb{L}_{1:T-1} = \sum_{t=2}^{T} D_{KL}(q( \boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)}) \Vert p_{\theta}(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)})) \tag{1.2.023}
$$
反向解码损失函数：
$$
\mathbb{L}_{0}= - \log{ p_{\theta}(\boldsymbol{x}_{(0)} \vert \boldsymbol{x}_{(1)}) } \tag{1.2.024}
$$
在式(1.2.021)中给定$\boldsymbol{x}_{(t)}$和$\boldsymbol{x}_{(0)}$时$\boldsymbol{x}_{(t-1)}$的概率可以定义为：
$$
q(\boldsymbol{x}_{(t-1)} \vert \boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)}) = \mathcal{N}(\boldsymbol{x}_{(t-1)}; \tilde{\boldsymbol{\mu}}_{t}(\boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)}, \tilde{\beta}_{t}\mathbb{I})) \tag{1.2.025}
$$
其中：
$$
\tilde{\boldsymbol{\mu}}_{t}(\boldsymbol{x}_{(t)}, \boldsymbol{x}_{(0)})=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}\boldsymbol{x}_{(0)} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}}\boldsymbol{x}_{(t)} \tag{1.2.026}
$$
$$
\tilde{\beta_{t}} = \frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t}}}\beta_{t} \tag{1.2.027}
$$
### 1.2.1. 损失函数讨论
#### 1.2.1.1. 损失函数$\mathbb{L}_{T}$
对于$\mathbb{L}_{T}$，由于我们忽略了$\beta_{t}$可以能过重参数化学习的特性，设其为常数，则：
$$
q(\boldsymbol{x}_{(t)} \vert \boldsymbol{x}_{(t-1)}) := \mathcal{N}(\boldsymbol{x}_{(t)}; \sqrt{1-\beta _{t}}\boldsymbol{x}_{(t-1)}, \beta _{t}\mathbb{I}), \quad t = 0, 1, 2, ..., T-1, T \tag{1.2.002}
$$
其中就没有可学习的参数，因此可以视为常量。
#### 1.2.1.2. 损失函数$\mathbb{L}_{1:T}$
对于：
$$
p_{\theta}(\boldsymbol{x}_{(t-1)}\vert \boldsymbol{x}_{(t)}) = \mathcal{N}(\boldsymbol{x}_{(t-1)};\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{(t)},t), \Sigma_{\theta}(\boldsymbol{x}_{(t)}, t)) \tag{1.2.005.1}
$$
我们首先设置方差$\Sigma_{\theta}(\boldsymbol{x}_{(t)},t)=\sigma_{t}^{2}\mathbb{I}$为常数。
其中当$\boldsymbol{x}_{(0)} \sim \mathbb{N}(\boldsymbol{0}, \mathbb{I})$时：
$$
\sigma_{t}^{2} = \beta_{t}
$$
当$\boldsymbol{x}_{(0)} \sim \mathbb{N}(\boldsymbol{v}, \mathbb{I})$时：
$$
\sigma_{t}^{2} = \tilde{\beta_{t}}=\frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t}}}\beta_{t}
$$
并且二者几乎没有差别。




# A. 附录
# A.1. 代码管理
```bash
ssh-keygen -t rsa -b 4096 -C "yt7589@qq.com"
# 保存在/home/psdz/diskc/yantao/github_id_rsa             Iching2020
chmod 600 /home/psdz/diskc/yantao/github_id_rsa
vim ~/.ssh/config
##############################################################################
Host github.com
  HostName github.com
  User git
  IdentityFile /home/psdz/diskc/yantao/github_id_rsa
##############################################################################
ssh -T yt7589@github.com
git remote set-url origin git@github.com:yt7589/mar.git
# 登录Github添加Deploy Key，将github_id_rsa.pub中内容贴到网页中
# 下载源码
git clone git@github.com:yt7589/mar.git
```

