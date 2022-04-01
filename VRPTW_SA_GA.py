import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from matplotlib.font_manager import FontProperties
import random

#遗传算法
class GA(object):
    def __init__(self,path_list, node_w, m, D, v, c, start_time, end_time, service_time):
        self.ga_choose_ratio = 0.2 #选择比例
        self.mutate_ratio = 0.05 #变异比例
        self.path_list=path_list #种群
        self.node_w=node_w #客户点需求量
        self.m=m #车辆数
        self.D=D #距离矩阵
        self.v=v #车速
        self.c=c #超载惩罚系数
        self.start_time=start_time #客户点左时间窗
        self.end_time=end_time #右时间窗
        self.service_time=service_time #服务时间
        self.iteration=10 #迭代次数
    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            cost = total_cost(fruit, self.node_w, self.m, self.D, self.v, self.c, self.start_time, self.end_time, self.service_time)  # 计算路径成本
            adp.append(1.0 / cost)
        return np.array(adp)

    #交叉
    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_conflict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_conflict_index.append(index)

        assert len(x_conflict_index) == len(y_conflict_index)

        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_conflict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)


    #选择父代候选
    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.path_list[index])
            parents_score.append(scores[index])
        return parents, parents_score

    #确定父代
    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])


    #变异：随机选择一个片段逆转
    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)


    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.path_list)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < 25:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / total_cost(gene_x_new, self.node_w, self.m, self.D, self.v, self.c, self.start_time, self.end_time, self.service_time)
            y_adp = 1. / total_cost(gene_y_new, self.node_w, self.m, self.D, self.v, self.c, self.start_time, self.end_time, self.service_time)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.path_list=fruits
        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None #最优个体
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1./best_score)
            #print(i,1./best_score)
        #print(1./best_score)
        return BEST_LIST,self.path_list

def paths_2opt(old_path):
    """
    产生新路径函数，本例中采用2-opt算法生成新路径
    :param old_path: 老路径
    :return: 新路径
    """
    N = len(old_path)
    a, b = np.random.randint(1, N-1), np.random.randint(1, N-1)  # 产生1-N之间的随机整数，这样可以保证路径的首尾为0
    random_left, random_right = min(a, b), max(a, b)  # 将生成的整数排序
    rever = old_path[random_left:random_right]  # 随机抽取的old_path中间部分的路径
    new_path = old_path[:random_left] + rever[::-1] + old_path[random_right:]  # 2-opt算法，翻转拼接成新路径
    #rever[::-1]相当于 rever[-1:-len(a)-1:-1],从最后一个元素到第一个元素复制一遍，即倒序

    return new_path

def heat(old_path, node_w, m, D, v, c, start_time, end_time, service_time):
    """
    加热函数
    :param old_path: 原始解，即初始化中设定的任意解
    :return: 返回初始温度T0和路径列表
    """
    dc = np.zeros(1000)  # 设定加热的次数，本例中使用2-opt算法对初始解进行数次的加热（随机打乱）
    path_list = []

    for i in range(100):
        for j in range(1000):
            new_path = paths_2opt(old_path)  #生成新路径
            cost1 = total_cost(old_path, node_w, m, D, v, c, start_time, end_time, service_time)  #计算老路径的距离
            cost2 = total_cost(new_path, node_w, m, D, v, c, start_time, end_time, service_time)  #计算新路径的距离
            dc[j] = abs(cost2 - cost1)  # 新老路径的距离偏差
            old_path = new_path
        path_list.append(new_path) #保存100条新路径至列表，作为遗传算法初始种群

    T0 = 0.1*max(dc)  # 将初始温度设置为最大偏差的0.1倍

    return T0, path_list

def new_paths(path_list, node_w, m, D, v, c, start_time, end_time, service_time):
    """
    用遗传算法产生新路径
    :param path_list: 路径列表（传入遗传的种群）
    :return: 新路径
    """
    #用2-opt算法生成新的种群成员，将种群规模恢复至100，提高种群多样性
    for i in range(2):
        for j in range(len(path_list)):
            new_path=path_list[j]
            for k in range(100):
                new_path=paths_2opt(new_path)
            path_list.append(new_path)

    GA_model = GA(path_list, node_w, m, D, v, c, start_time, end_time, service_time) #遗传算法生成新路径
    path_after_ga,path_list = GA_model.run()

    return path_after_ga,path_list


def total_cost(path, node_w, m, D, v, c, start_time, end_time, service_time):
    """
    计算距离函数，这里有一个小技巧，就是将距离提前计算好形成二维列表，以后只需要查找列表就可以计算距离，大大节省时间
    :param D: 节点之间的距离
    :param m: 车辆个数
    :param path: 待计算的路径
    :param node_w: （初始定义的）各个点的权重--即客户点的需求量
    :param v: 车辆速度
    :param c: 超载惩罚系数
    :param start_time: 起始时间窗
    :param end_time: 中止时间窗
    :param service_time: 服务时间
    :return: 目标函数，即当前路径的距离值+惩罚项综合
    """
    dis = 0
    w_waste = 1 #时间浪费的惩罚系数
    w_punish = 2 #超时的惩罚系数

    for i in range(len(path) - 1):  # 求解路径path中两点之间的距离，通过查找列表的方式来返回距离值
        dis += D[path[i]][path[i + 1]]
    dis+=D[0][path[0]]
    dis+=D[path[-1]][0]

    address_index = [i for i in range(len(path)) if path[i] <= 8]  # 查找路径中的仓库的坐标
    address_index.insert(0,0)#车辆从仓库出发，在仓库列表第一位添加一个仓库点
    #print("address_index:",address_index)
    C = [0] * m  # 每辆车的容量
    M = [0] * m  # 设置惩罚项，超过每辆车的最大容量200则进行惩罚

    for i in range(len(address_index) - 1):  # 仓库坐标之间的编号就是每辆车行驶的路线
        for j in range(address_index[i], address_index[i + 1], 1):
            C[i] += node_w[path[j]]  # 计算每辆车的当前容量，确保不能超过最大容量200的限制
        if C[i] >= 200:
            M[i] = c * (C[i] - 200)  # 惩罚项，防止车辆的容量超过最大限度200，惩罚项系数20可以修改
    #计算最后一辆车的超重成本
    for a in range(address_index[-1]+1,len(path)):
        C[-1]+=node_w[path[a]]
        if C[-1]>=200:
            M[-1]=c*(C[-1] - 200)
    #print("C:",C)

    #时间窗
    time_waste = [0] * m #定义一个列表储存每辆车的时间浪费成本
    time_punish = [0] * m #储存每辆车的超时成本
    total_t = [0] * m #每辆车所用总时长
    for i in range(len(address_index)-1):
        for j in range(address_index[i], address_index[i+1], 1):
            total_t[i]+=D[path[j]][path[j+1]]/v
            if total_t[i]<start_time[path[j+1]]:
                total_t[i]=start_time[path[j+1]]
                time_waste[i]+=w_waste*(start_time[path[j+1]]-total_t[i])
            elif total_t[i]>end_time[path[j+1]]:
                time_punish[i]+=w_punish*(total_t[i]-end_time[path[j+1]])
            total_t[i]+=service_time[path[j+1]]
    #计算最后一辆车的时间成本
    for b in range(address_index[-1]+1,len(path)-1):
        total_t[-1]+=D[path[b]][path[b+1]]/v
        if total_t[-1]<start_time[path[b+1]]:
            total_t[-1]=start_time[path[b+1]]
            time_waste[-1]+=w_waste*(start_time[path[b+1]]-total_t[-1])
        elif total_t[-1]>end_time[path[b+1]]:
            time_punish[-1]+=w_punish*(total_t[-1]-end_time[path[b+1]])
        total_t[-1]+=service_time[path[b+1]]
    #print("total_t:",total_t)
    #print("time_waste:",time_waste)
    #print("time_punish:",time_punish)

    sum_cost = sum(M) + sum(time_waste) + sum(time_punish)
    total=dis+sum_cost
    #print("sum M=",sum(M))
    #print("total cost=",total)

    return total


def init(n, m):
    """
    初始化函数，包括读取坐标数据，设定初始路径等
    :param n: n为客户数量
    :param m: m为车辆数量
    :return: 返回初始路径init_path、节点的x坐标node_X、节点的y坐标node_y、节点的权重(需求量)node_w、车辆数量m、 节点时间窗开始时间start_time、 节点时间窗结束时间end_time、 节点服务时间service_time
    """

    def get_data():
        """
        利用xlrd模块来读取excel中的某一列的数据并返回
        """
        data = xlrd.open_workbook('data_timewindows_SA_GA.xlsx')  # 打开选择的excel表格；节点0-8共九个节点均代表仓库
        table = data.sheets()[0]  # 读取excl中的某张表单，0代表默认首选表单
        return table.col_values(1), table.col_values(2), table.col_values(3), table.col_values(4), table.col_values(5), table.col_values(6)

    node_x, node_y, node_w , start_time, end_time, service_time= get_data()  # 读取仓库和商店的坐标点和需求量信息
    print("node_w:",node_w)

    D = [[0] * len(node_x) for i in range(len(node_x))]  # 定义二维列表，用于保存节点之间的距离
    for i in range(len(node_x)):
        for j in range(len(node_x)):
            D[i][j] = np.sqrt((node_x[i] - node_x[j]) ** 2 + (node_y[i] - node_y[j]) ** 2)  # 计算两点间距离

    #生成初始解
    init_path=[]
    for i in range(len(node_x)):
        init_path.append(i)
    print("init_path:", init_path)
    print("length of init_path:", len(init_path))
    return init_path, D, node_x, node_y, node_w, start_time, end_time, service_time, m


def metropolis(old_path, new_path, node_w, m, T, D, v, c, start_time, end_time, service_time):
    """
    metropolis准则，即是模拟退火算法中接受新解的概率，为防止陷入局部最优解，因此需要以一定概率接受新解
    :param old_path: 老路径
    :param new_path: 新路径
    :param node_w: 商店的货物需求量
    :param m: 车辆数量
    :param T: 模拟退火外循环下的当前温度
    :param D: 节点之间的距离
    :return: 返回metropolis准则判断下的当前最优解和对应的目标函数（距离）
    """
    cost1 = total_cost(old_path, node_w, m, D, v, c, start_time, end_time, service_time)  # 老路径的距离
    cost2 = total_cost(new_path, node_w, m, D, v, c, start_time, end_time, service_time)  # 新路径的距离

    dc = cost2 - cost1  # 二者差值

    if dc < 0 or np.exp(-abs(dc) / T) > np.random.random():  # metropolis准则，接受新解的两种情况：1.新解的距离更小；2.一定概率接受; np.random.random():(0,1)之间的随机数
        path = new_path
        path_cost = cost2
    else:
        path = old_path
        path_cost = cost1

    return path, path_cost


def CVRP_SA():
    init_path, D, node_x, node_y, node_w, start_time, end_time, service_time, m = init(100, 10)  # 本例中共有商店100个、仓库1个和车辆10辆

    v = 1  # 车辆速度
    c = 20000  # 超载惩罚系数

    T0, path_list = heat(init_path, node_w, m, D, v, c, start_time, end_time, service_time)  # 初始温度，（加热过程后的路径作为）初始路径
    print("加热后的初始解为：",path_list)
    print("初始温度为：", T0)

    T_down_rate = 0.9  # 温度下降速率
    T_end = 0.01  # 终止温度
    K = 5  # 内循环次数

    count = math.ceil(math.log(T_end / T0, T_down_rate))  # 外循环次数
    print("count=",count)
    dis_T = np.zeros(count + 1)  # 每次循环下的最优解
    best_path = init_path  # 设置最优路径为初始路径
    shortest_dis = np.inf  # 设置初始最优距离为无穷
    n = 0
    T = T0  # 当前循环下的温度值

    old_path,path_list = new_paths(path_list, node_w, m, D, v, c, start_time, end_time, service_time)
    while T > T_end:  # 外循环：模拟退火直到温度小于终止温度
        for i in range(K): #内循环
            new_path,path_list = new_paths(path_list, node_w, m, D, v, c, start_time, end_time, service_time)  # 生成新路径
            old_path, path_dis = metropolis(old_path, new_path, node_w, m, T, D, v, c, start_time, end_time, service_time)  # 通过metropolis准则判断当前最优路径
            if path_dis <= shortest_dis:  # 如果最优路径比之前计算的结果更好，就接受
                shortest_dis = path_dis
                best_path = old_path

        dis_T[n] = shortest_dis  # 每次循环下的最优解
        n += 1
        T *= T_down_rate  # 每次循环后，以一定的下降速率降温
        print("best_path of this epoch:",best_path)
        print('best_cost', shortest_dis)  # 将执行过程print出来，可以时刻监控T值和当前的最优距离
        print("T=",T)

    print('best_path', best_path)  # 输出最优的路径解
    print('退火算法计算CVRP问题的结果为', total_cost(best_path, node_w, m, D, v, c, start_time, end_time, service_time))  # 输出最优的计算结果，即车辆的形式路径总和

    best_path.append(0)
    best_path.insert(0, 0)
    address_index = [i for i in range(len(best_path)) if best_path[i] <= 8]  # 查找路径中的仓库的坐标

    for i in range(len(address_index) - 1):  # 输出每辆车的路径
        print('第{}辆车的路径为：'.format(i + 1))
        print(best_path[address_index[i]:address_index[i + 1] + 1])

    picture(node_x, node_y, best_path)  # 车辆访问路径可视化

def picture(node_x, node_y, best_path):
    """
    绘制车辆的访问路径图函数
    :param node_x: 商店的位置x坐标
    :param node_y: 商店的位置y坐标
    :param best_path: 最优路径
    :return:
    """

    def background():
        """
        利用plt画出坐标系背景，包括设置坐标和尺寸等信息
        """
        font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)  # 从系统中调取中文字体，方便plt展示中文标签
        plt.figure(figsize=(10, 10))  # 设置图片的尺寸，（长，宽），单位是英寸
        plt.xlim((0, 100))  # 设置x轴的数值显示范围
        plt.ylim((0, 100))  # 设置y轴的数值显示范围
        plt.xticks(np.linspace(0, 100, 11))  # 设置x轴的刻度，其中0、100代表x轴左右极限值，11代表将x轴分成10份，进行标记
        plt.yticks(np.linspace(0, 100, 11))  # 设置y轴的刻度，其中0、100代表y轴左右极限值，11代表将y轴分成10份，进行标记
        plt.xlabel('x坐标-示意', fontproperties=font)  # 设置x轴显示的标签，效果为中文
        plt.ylabel('y坐标-示意', fontproperties=font)  # 设置y轴显示的标签，效果为中文

    def random_color():
        """产生随机的十六进制颜色返回"""
        colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        color = ""  # 定义color是字符串型数据
        for i in range(6):
            color += colorArr[np.random.randint(0, 14)]
        return "#" + color

    x_list = []  # 定义plt画图的节点x坐标
    y_list = []  # 定义plt画图的节点y坐标
    background()

    plt.scatter(node_x, node_y, c='r', s=50, alpha=1)  # 画出散点图，即客户点
    address_index = [i for i in range(len(best_path)) if best_path[i] <= 8]  # 查找路径中的0（仓库）的坐标
    address_index.insert(0,0)

    for i in range(len(address_index) - 1):
        for j in range(address_index[i], address_index[i + 1] + 1, 1):
            x_list.append(node_x[best_path[j]])
            y_list.append(node_y[best_path[j]])
        plt.plot(x_list, y_list, c=random_color())  # 绘制每辆车的路径
        x_list = []  # 清空
        y_list = []  # 清空

    plt.savefig('result.png')  # 保存车辆访问路径可视化图
    plt.show()


if __name__ == '__main__':
    start_datetime = datetime.datetime.now()
    CVRP_SA()
    end_datetime = datetime.datetime.now()
    print('程序运行时间为：', end_datetime - start_datetime)
