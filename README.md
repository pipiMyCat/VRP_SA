# VRP_SA
练习时长四天半的练习生使用模拟退火算法（SA）解决VRP问题的一些小尝试
项目中包括：
CVRP_SA：模拟退火解决带容量约束的VRP问题；初始解：随机；新路径生成：2-opt 【对应数据集：Data0/Data】

VRPTW_SA：模拟退火解决带软时间窗和容量约束的VRP问题；初始解：随机；新路径生成：2-opt 【对应数据集：data_timewindows】

VRPTW_SA_pro：模拟退火解决带软时间窗和容量约束的VRP问题；初始解：贪婪+随机；新路径生成：2-opt【对应数据集：data_timewindows】

VRPTW_SA_pro2：模拟退火解决带软时间窗和容量约束的VRP问题；初始解：仅贪婪；新路径生成：2-opt【对应数据集：data_timewindows】

VRPTW_SA_GA：模拟退火+遗传解决带软时间窗和容量约束的VRP问题；初始解：随机；新路径生成：遗传算法【对应数据集：data_timewindows_SA_GA】

其中CVRP_SA代码来源于JIANGFS/CVRP-Simulated-Annealing-Algorithm项目，感谢大佬(>_<)

当前代码仍存在一些问题：
收敛性较差；
运行时间较长；
对结果作图时可能出现小问题

以上代码还有较大改进空间，甚至可能存在错漏，仅供参考，欢迎交流学习(>_<)
