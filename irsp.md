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
我们的模型要使$\boldsymbol{x}_{(0)}$出现的概率最大，因此定义损失函数为：
$$
\mathcal{L}_{f} = \mathbb{E}\left[ -\log{p_{\theta}(\boldsymbol{x}_{(0)})} \right] \tag{1.2.003}
$$
我们可以将上式视为分母为1的式子，我们以概率替换$q(\boldsymbol{x}_{(1:T)}\vert \boldsymbol{x}_{(0)})$，由于概率值小于1，故有：
$$
\mathcal{L}_{f}=\mathbb{E}\left[ -\log{p_{\theta}(\boldsymbol{x}_{(0)})} \right]\le \mathbb{E}_{q}\left[ -\log{\frac{p_{\theta}(\boldsymbol{x}_{(0:T)})}{q(\boldsymbol{x}_{(1:T)}\vert \boldsymbol{x}_{(0)})}} \right] \tag{1.2.004}
$$
下面我们来研究$p_{\theta}(\boldsymbol{x}_{(0)})$，其代表在我们以$\theta$为参数的模型下，通过不断向$\boldsymbol{x}_{(0)}$



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

