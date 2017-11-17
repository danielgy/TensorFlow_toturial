# 并发与并行
---

并发 VS 并行：
       首先并发**不等于**并行，盗用一张Erlang之父Joe Armstrong画的图：
       ![并发VS并行](https://github.com/danielgy/TensorFlow_toturial/blob/master/python/image/concurrent%26paralism.jpg)
       
---
       并发：不要求“同时”进行（看起来像多任务）
       并行：要求“同时”进行（真正的多任务）

###  Python解决之道

Python用于解决并行和并发问题的方法包括 多进程、多线程、concurrent包、Gevent、Celery等。方法的选择基于任务（IO密集型 or CPU密集型）和开发方式(事件驱动的协作多任务 or 优先多任务处理)。最常用的是多进程和多线程，区别如下表所示：
#### **多进程VS多线程**
| process |thread |
| ------------- |:-------------:|
| 不共享内存| 共享内存 |
| 切换代价大 | 切换代价不大 | 
| 需要更多的资源 | 需要相对少的资源，轻量级的进程 |
|不需要内存同步|需要内存同步机制确保正确访问数据|
|CPU密集型|IO密集型|
	
	Concurrent是基于threads的high-level API. Gevent也是一个解决高并发的常用选择之一，其执行顺序是确定的，在一个函数运行时，中断函数并保存函数上下文，之后运行其他函数，之后再返回原函数继续执行。Celery是Python开发的分布式任务调度模块，其核心是以任务为单元进行执行，Celery为运行任务提供了很大的灵活性:支持同步或异步执行，在同一台机器上或在多台机器上可以使用线程、进程、Eventlet或gevent。


### 实践

以访问数据库为例，设置worker=4，不同方式下的代码和结果运行如下：

---
**Thread：**

```python
2017-11-17 14:37:40   Worker  1 :    Init  and Start Running.. [Pid: 17176]
2017-11-17 14:37:40   Worker  2 :    Init  and Start Running.. [Pid: 17176]
2017-11-17 14:37:40   Worker  3 :    Init  and Start Running.. [Pid: 17176]
2017-11-17 14:37:40   Worker  4 :    Init  and Start Running.. [Pid: 17176]
2017-11-17 14:37:40  Worker Manager:   start Workers  [ Worker: 4][JobInQueue: 3] . 
'PID' 'result' 
17176 (0, 10)
17176 (0, 10)
17176 (0, 10)
17176 (0, 10)
17176 (0, 10)
17176 (0, 10)
17176 (0, 10)
2017-11-17 14:37:49[Pid: 17176] Worker  3 :   Finish All Job   and Exit    . 
2017-11-17 14:37:52[Pid: 17176] Worker  2 :   Finish All Job   and Exit    . 
2017-11-17 14:37:52[Pid: 17176] Worker  1 :   Finish All Job   and Exit    . 
2017-11-17 14:37:52[Pid: 17176] Worker  4 :   Finish All Job   and Exit    . 
2017-11-17 14:37:52  Worker Manager:  All Worker Process Finish[ Cost: 12.17  sec]. 
```

**Process**
```python
2017-11-17 14:39:04  Worker Manager:   start Workers  [ Worker: 4][JobInQueue: 7] . 
2017-11-17 14:39:05   Worker  1 :    Init  and Start Running.. [Pid: 6632]
2017-11-17 14:39:05   Worker  2 :    Init  and Start Running.. [Pid: 16400]
2017-11-17 14:39:05   Worker  3 :    Init  and Start Running.. [Pid: 16812]
2017-11-17 14:39:05   Worker  4 :    Init  and Start Running.. [Pid: 23780]
'PID' 'result' 
16400 (0, 10)
6632 (0, 10)
16812 (0, 10)
23780 (0, 10)
16400 (0, 10)
6632 (0, 10)
16812 (0, 10)
2017-11-17 14:39:15[Pid: 23780] Worker  4 :   Finish All Job   and Exit    . 
2017-11-17 14:39:17[Pid: 16400] Worker  2 :   Finish All Job   and Exit    . 
2017-11-17 14:39:17[Pid: 6632] Worker  1 :   Finish All Job   and Exit    . 
2017-11-17 14:39:18[Pid: 16812] Worker  3 :   Finish All Job   and Exit    . 
2017-11-17 14:39:19  Worker Manager:  All Worker Process Finish[ Cost: 14.94  sec]. 
```


**Concurrent**
```python
'PID' 'result' 
11588 (0, 10)
11588 (0, 10)
11588 (0, 10)
11588 (0, 10)
11588 (0, 10)
11588 (0, 10)
11588 (0, 10)
Finish[ Cost: 6.14  sec]
```

**Gevent**
```python
'PID' 'result' 
24632 (0, 10)
24632 (0, 10)
24632 (0, 10)
24632 (0, 10)
24632 (0, 10)
24632 (0, 10)
24632 (0, 10)
Finish[ Cost: 21.23  sec]
```

**Celery**
![celery](https://github.com/danielgy/TensorFlow_toturial/blob/master/python/image/celery.jpg)

