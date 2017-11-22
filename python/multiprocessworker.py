
from multiprocessing import Pool,Queue,Process
import os
 
import time
import datetime
# import pub.pubutils as pubutils
 


###公共函数 获取时间
def get_time_string(cdatetime=None,indayformat=False):
    """ 获得时间的标准字符串   到 天 或者 秒
      
    Args:
        cdatetime  需要转换的时间  如果为None 则代表当前时间
        indayformat:  是否为日期格式   如果true 则为%Y-%m-%d  ，否则到秒  %Y-%m-%d %H:%M:%S  默认False
        
    Returns:
        转换后的字符串结果  如果原来时间为None  则返回当前时间的字符串
    """
    ctime=cdatetime
    if ctime is None:
        ctime=datetime.datetime.now()
    
    if indayformat==True:
        return ctime.strftime('%Y-%m-%d' )
    else:
        return ctime.strftime('%Y-%m-%d %H:%M:%S' )

# 工作进程的配置  
class workConfig(object):    
    worker_num=3 
    worker_name=[]    
    show_runtime=False
    job_retry_times=3 
       
  

class workManager(object):
    def __init__(self, work_config,work_queue):
        self.work_config=work_config
        self.work_queue = work_queue        
        self.workers = []
        
        self.start_time = time.time()
         
        #self.__init_worker_pool(self.work_config.worker_num)
        

    """
        初始化进程线程
    """
    def start_work(self ):
        
        #pool =  Pool(worker_num)        
        
        for i in range(0,self.work_config.worker_num):
            #pool.apply_async( workerProcess, args=(self, self.work_config,' Worker  '+str(i)))
            p = Process(target=workerProcess, args=(self, self.work_config,' Worker  '+str(i+1)))
            p.start()
            self.workers.append(p)
            
         
        print(get_time_string()+"  Worker Manager:   start Workers  [ Worker: " +str(self.work_config.worker_num)+"][JobInQueue: "+str(self.work_queue.qsize())+"] . \n\n" )

        self.workers[self.work_config.worker_num-1].join()
        
        #pool.close()
        #pool.join()
 
        self.allworker_finished()
        
    """
         所有进程运行完毕  
    """   
    def allworker_finished(self):
#        for item in self.threads:
#           if item.isAlive():
#              item.join()
         
        while True:
            finishAll=True
            for i in range(0,len( self.workers)):
                p=self.workers[i]
                #print(p.exitcode)
                if p.exitcode==None   :  #还在运行
                    finishAll=False
            if finishAll==True :
                break      
            else :
                time.sleep(2)  
                 
        timecost=format(time.time()-  self.start_time,'.2f' ) ;
        print("\n\n"+get_time_string()+"  Worker Manager:  All Worker Process Finish[ Cost: "+str(timecost)+"  sec]. \n\n" )
        

#工作进程  实际执行相关的业务处理
def workerProcess( work_manager,work_config,worker_name):
      
        #if work_manager.showruntime:
        print(get_time_string()+'  '   +worker_name+' :    Init  and Start Running.. [Pid: '+str(os.getpid())+']' )
                
        retry_times=0;
        jobnum=0
        while True :
            if work_manager.work_queue.qsize()<=0 :   #如果队列没有任务了  继续尝试 最大次数后退出
                if work_config.job_retry_times> 0    :   #如果禁止重新尝试  或者尝试提取任务次数达到最大值
                    while  retry_times< work_config.job_retry_times : 
                        retry_times+=1
                        #print(worker_name+' :  Sleep 3s.  retry: ' ,retry_times)
                        time.sleep(3)  
                break  
            
               
            try:                
                if  work_config.show_runtime:
                    print('['+str(os.getpid())+']'+worker_name  +'  :  Step 2   Get  New job from queue [Size: '    +str(work_manager.work_queue.qsize())+']')
                
                processfunc, args = work_manager.work_queue.get()#任务异步出队，Queue内部实现了同步机制
                
                if work_config.show_runtime:
                    print('['+str(os.getpid())+']'+worker_name+' :  Step 3 Process  job... [Total ' +str(jobnum)+']' )
                jobnum+=1    
                processfunc(args)
                                
                           
            except Exception as e:
                print('['+str(os.getpid())+']'+worker_name  +' :  Raise Exception  :' ,e)
                #break
        
       # if  work_config.show_runtime:
        print( get_time_string()+'[Pid: '+str(os.getpid())+']'+ worker_name +' :   Finish All Job   and Exit    . '  )                      






###多进程的处理模式的支持函数    
def process_with_multitworker(workqueue,worker_num):
             
    # 1 初始化运行配置    
    workconfig = workConfig()
    workconfig.worker_num=worker_num
    #workconfig.show_runtime=False
    
              
    #3初始化线程池 开始工作
    work_manager =  workManager(workconfig,workqueue)
    #4 等待工作完成   
    work_manager.after_allworker_complete()
    







# 模拟测试程序
def test_job(args):     
    time.sleep(3)#模拟处理时间
    print(str(os.getpid()), list(args))
  
        

if __name__ == '__main__':
       
  
    
    
    # 1 初始化运行配置    
    workconfig = workConfig()
    workconfig.worker_num=6      #工作进程数  不要超过cpu核数

    workconfig.job_retry_times=2  # 没有任务时的最大尝试提取任务的次数  如为0 则直接退出
    workconfig.show_runtime=True   #是否显示运行时状态信息 默认False 
    
            
    #2初始化工作任务队列
    workqueue= Queue();      
    
    #将工作函数和参数全传入
    for i in [1,2,3,4,5,6,7,8] :
        args=(i,'test',3)
        workqueue.put((test_job, args))  #将任务和参数对象穿进去队列 
     
          
    #3 根据配置初始化多个进程   并开始工作
    work_manager =  workManager(workconfig,workqueue)
    work_manager.start_work()
    
                
