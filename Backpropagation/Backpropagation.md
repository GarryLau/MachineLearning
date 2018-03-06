# 反向传播Backpropagation
<div align=center><img src="http://img.blog.csdn.net/20180306160755211?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 373 height = 220 alt="ellipse" align=center /></div><div align=center>图1 </div>
在构造神经网络的时候我们需要知道如何进行训练。反向传播是常用的用来训练神经网络的技术。
## 总览
本文以一个三层的神经网络来说明反向传播：

+ 有两个神经元的输入层
+ 有两个神经元的隐层
+ 有单个神经元的输出层
<div align=center><img src="http://img.blog.csdn.net/20180306162409608?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 651 height = 365 alt="ellipse" align=center /></div><div align=center>图2 </div>
## 权重
神经网络的训练就是找到最佳的权重使得预测误差最小。权重通常用一些随机数来进行初始化。然后再用反向传播不断地更新权重使得预测值和真实值的差距越来越小。
本文我们给权重的初始化数据为：<font color=blue>w1 = 0.11</font>, <font color=blue>w2 = 0.21</font>, <font color=blue>w3 = 0.12</font>, <font color=blue>w4 = 0.08</font>, <font color=blue>w5 = 0.14</font>, <font color=blue>w6 = 0.15</font>。
<div align=center><img src="http://img.blog.csdn.net/20180306162729569?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 662 height = 364 alt="ellipse" align=center /></div><div align=center>图3 </div>
## 数据集
我们的输入数据是两维的，输出数据是一维的。
<div align=center><img src="http://img.blog.csdn.net/201803061849014?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 351 height = 152 alt="ellipse" align=center /></div><div align=center>图4 </div>
我们示例使用的输入数据是<font color=blue>inputs=[2,3]</font>，输出数据是<font color=blue>output=[1]</font>。
<div align=center><img src="http://img.blog.csdn.net/20180306185035290?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 351 height = 152 alt="ellipse" align=center /></div><div align=center>图5 </div>
## 前向传播过程
我们根据给定的输入去预测输出。利用输入乘以对应的权重，得到中间结果之后，再用中间结果乘以后续的权重，如此进行直到输出。
<div align=center><img src="http://img.blog.csdn.net/20180306190030736?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 672 height = 538 alt="ellipse" align=center /></div><div align=center>图6 </div>
## 计算误差
紧接着我们计算网络的预测值（输出）与真实值之间的差异。如图7可知，目前的预测值与输出值相差甚远。图7展示了误差的计算方法。
<div align=center><img src="http://img.blog.csdn.net/20180306190614284?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 824 height = 484 alt="ellipse" align=center /></div><div align=center>图7 </div>
## 如何减少误差
我们的目标是通过训练减少预测值与真实值之间的差异。由于真实值是不变的，因此减少误差的唯一途径就是改变预测值。现在的问题就是，如何去改变预测值？通过分析前向运算的预测过程我们可知预测值是权重的函数。想要改变预测值我们需要改变权重值。
<div align=center><img src="http://img.blog.csdn.net/20180306191117394?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 669 height = 308 alt="ellipse" align=center /></div><div align=center>图8 </div>
现在的问题是如何改变（更新）权重来减少预测值与真实值之间的误差呢？
答案就是反向传播！
## 反向传播
反向传播（即误差的反向传播）是用梯度下降法更新权重的一种机制。反向传播需要计算损失函数对权重的导数。这种求导是贯穿整个神经网络的。
> 梯度下降是一种迭代性的优化算法，用以找到函数的最小值；在这里我们需要通过梯度下降最小化损失函数。利用梯度下降找最小值的过程需要更新权重，权重的更新方法如图9所示：
<div align=center><img src="http://img.blog.csdn.net/20180306191926600?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 414 height = 267 alt="ellipse" align=center /></div><div align=center>图9 </div>
比如，为了更新<font color=blue>w6</font>，我们需要用<font color=blue>w6</font>减去损失函数对<font color=blue>w6</font>的导数。一般情况下我们还需要用该导数乘以一个超参数——学习率<font color=Orange>a</font>。
<div align=center><img src="http://img.blog.csdn.net/20180306192729735?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 375 height = 97 alt="ellipse" align=center /></div><div align=center>图10 </div>
损失函数的求导遵循链式法则：
<div align=center><img src="http://img.blog.csdn.net/20180306192915224?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 1056 height = 402 alt="ellipse" align=center /></div><div align=center>图11 </div>
由图11的推导我们可知，<font color=blue>w6</font>可由下式进行更新：
<div align=center><img src="http://img.blog.csdn.net/20180306193317210?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 375 height = 97 alt="ellipse" align=center /></div><div align=center>图12 </div>
类似的<font color=blue>w5</font>可由下式进行更新：
<div align=center><img src="http://img.blog.csdn.net/20180306193406340?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 375 height = 102 alt="ellipse" align=center /></div><div align=center>图13 </div>
现在还需要对输入层和隐层之间的<font color=blue>w1</font>，<font color=blue>w2</font>，<font color=blue>w3</font>，<font color=blue>w4</font>进行更新。损失函数对<font color=blue>w1</font>的偏导如下：
<div align=center><img src="http://img.blog.csdn.net/20180306193722637?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 938 height = 372 alt="ellipse" align=center /></div><div align=center>图14 </div>
类似于<font color=blue>w1</font>的更新方式，我们同样的可对<font color=blue>w2</font>，<font color=blue>w3</font>，<font color=blue>w4</font>进行更新：
<div align=center><img src="http://img.blog.csdn.net/20180306193904572?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 518 height = 288 alt="ellipse" align=center /></div><div align=center>图15 </div>
可以以矩阵乘法的形式重新描述上述更新公式：
<div align=center><img src="http://img.blog.csdn.net/20180306194024497?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 758 height = 155 alt="ellipse" align=center /></div><div align=center>图16 </div>
## 反向传播过程
利用上面推导出的公式我们可以计算出新的权重。
> 学习率是可以通过交叉验证得到的超参数。
<div align=center><img src="http://img.blog.csdn.net/20180306194254754?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 932 height = 267 alt="ellipse" align=center /></div><div align=center>图17 </div>
现在用更新后的权重重新计算前向传播：
<div align=center><img src="http://img.blog.csdn.net/20180306194430968?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 701 height = 550 alt="ellipse" align=center /></div><div align=center>图18 </div>
由图18可知现在的预测值是<font color=blue>0.26</font>，比之前预测的<font color=blue>0.191</font>更靠近真实值。我们可以重复上述过程直到预测值和真实值之间的差异为零。

## 参考文献
[Backpropagation Step by Step](http://hmkcode.github.io/ai/backpropagation-step-by-step/)
<div align=center><img src="http://img.blog.csdn.net/2018030517011018?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Z2FuNTI4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width = 384 height = 384 alt="ellipse" align=center /></div><div align=center></div>
更多资料请移步github： 
https://github.com/GarryLau/MachineLearning
