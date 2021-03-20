# MyPaGraph
An Examination For PaGraph

这是对PaGraph的一些基本实验，

核心文件为：

| 文件名       | 描述                                       |
| ------------ | ------------------------------------------ |
| MyPaGraph.py | PaGraph 模块，包括图划分算法和存储查询模块 |
| TrainMul.py  | 多进程多GPU训练                            |
| TrainSin.py  | 单进程单GPU训练                            |



------

MyPaGraph.py

```python
#以下用例来源：Test3.py

MyPaGraph.Init.DivideGraph(g,num,hop,target_nodes = None,
                           PV_need = True,print_info=True,information_print_gap = -1)
'''
描述:执行图划分算法，按照PaGraph论文中的算法将DGL图进行划分。

参数:
g:	DGLGraph对象，是需要划分的图。
num:	int,需要划分图的数量，一般为gpu的个数。
hop:	int,划分图节点的跳数，一般为2。
target_nodes:	list,需要划分的节点，None代表全部划分。这等同于论文中TV的并集。
PV_need:	bool,是否需要论文中所描述的PV，默认需要。
print_info:	bool,是否打印相关信息，默认为是。
information_print_gap:	int,每划分多少个点打印一次信息，默认为每划分10%节点打印一次。

返回值:
TV,PV  (if PV_need)
TV     (if not PV_need)

TV,PV意义如论文所指。这里TV,PV是一个list，分别存放了每个子图真正的TV和PV。

例：
import MyPaGraph as pg
import torch
import numpy as np
from dgl.data import RedditDataset
data = RedditDataset()
g = data[0]
TV,PV=pg.Init.DivideGraph(g=g,num=2,hop=1)

'''


MyPaGraph.Storage(g,data,cache_rate,nodes,gpu,cpu='cpu')
'''
描述:创建Storage对象，按照cache_rate将数据存储在gpu和cpu中。

参数:
g:	DGLGraph对象。训练的图。
data:	需要存储的信息的字典，如{'feat':[feat对应的tensor],'label':[label对应的tensor]},一般直接传入g.ndata即可。
cache_rate:	float,数据缓存到gpu中的比率。
nodes:	tensor,需要存储的总节点，这一般为相应子图的PV。
gpu:	str,存储数据的gpu，如'cuda:0'或'cuda:1'。
cpu:	str,一般存储数据的主机（代表次优先级的设备，gpu为最高优先级设备），默认为'cpu'

返回值:
None

例：
st = pg.Storage(g=g,data=g.ndata,gpu='cuda:0',cache_rate = 0.1,nodes=PV[0])

'''

MyPaGraph.Storage.Query(fname,nodes,print_info=False)
'''
描述:在一个Storage对象里，查询目标节点的相关特征的tensor。

参数:
fname:	str，要查询的特征的名字，如'feat','label'
nodes:	tensor，要查询的目标节点的张量，注意这里的节点应为创建对象时传参nodes的子集。
print_info:		bool,是否打印相关信息，默认为否。

返回值:
result:		tensor，目标节点的对应特征的张量，处于gpu上。

例：
qn1 = torch.arange(140,5,-1)
r1 = st.Query(fname='label',nodes=qn1,print_time=True)

'''


```



运行方法：

python TrainMul.py [参数表，如--ngpus=2等]

python TrainSin.py [参数表，如--data='reddit'等]

参数表详见代码。



例：

python3 TrainMul.py --n-epochs=10 --cache-rate=0.24 --batch-size=256 --ngpus=-1

python3 TrainMul.py --n-epochs=10 --cache-rate=0.8  --batch-size=240 --ngpus=3 --data='pubmed'

python TrainMul.py --n-epochs=100 --cache-rate=0.8  --batch-size=240 --ngpus=2 --data='pubmed'