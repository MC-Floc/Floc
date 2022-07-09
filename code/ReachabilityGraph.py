# -*- coding: utf-8 -*-
from Parameter import Parameter as Para

# 常规提取方法
# Input：数据所在文件夹地址（OneLabelData）
# Output：状态可达图及其基本信息
def graph_extract(folder_path):
    if folder_path == "":
        filepath_rg = folder_path + "rg.txt"           # 状态可达图
        filepath_vs = folder_path + "vs.txt"           # 违例状态
        filepath_vs_l = folder_path + "vs_l.txt"           # 违例状态
    else:
        filepath_rg = folder_path + "/rg.txt"           # 状态可达图
        filepath_vs = folder_path + "/vs.txt"           # 违例状态
        #filepath_label = folder_path + "/label.txt"     # 标记
        filepath_vs_l = folder_path + "/vs_l.txt"           # 违例状态

    graph_temp = [line.strip() for line in open(filepath_rg, 'r', encoding='utf-8').readlines()]
    violation_temp = [line.strip() for line in open(filepath_vs, 'r', encoding='utf-8').readlines()]
    violation_l_temp = [line.strip() for line in open(filepath_vs_l, 'r', encoding='utf-8').readlines()]
    #label_temp = [line.strip() for line in open(filepath_label, 'r', encoding='utf-8').readlines()]
    # Get trace_num and state_num
    graph_temp[0] = graph_temp[0].replace('des', '')
    val = eval(graph_temp[0])                     # 把字符串转为数组
    edge_num = val[1]                            # 边数
    state_num = val[2]                            # 节点数
    graph_temp.remove(graph_temp[0])
    #print(val)
    #print(edge_num)

    # Get Reachability Graph
    graph = {}                        # eg: {0: [(2, 'MakeChoice.t1'), (3, 'PutOfferRequest.t1')]} map: int 双层数组
    converse_graph = {}
    for i in range(state_num):
        graph[i] = []
        converse_graph[i] = []
    transition2num = {}                        # map {'p_start': 0, 'r_starts': 1}
    # num2transition = {}
    i = 0
    for edge in graph_temp:
        val = eval(edge)                    # (0,"P0_0",1)
        graph[val[0]].append((val[2], val[1]))
        converse_graph[val[2]].append((val[0], val[1]))

        if not val[1] in transition2num:
            transition2num[val[1]] = i
            # num2transition[i] = val[1]
            i += 1

        # print(transition2num)  #记录了所有有向图的某个节点出发的节点路径等

    # start = [state[0] for state in graph[0]]
    # start.append(0)
    violation = list(map(int, violation_temp[0].split(" ")))        # eg:[105] 违例状态
    violation_l=[]
    if len(violation_l_temp) != 0:
        violation_l = list(map(int, violation_l_temp[0].split(" ")))
    label = 0#transition2num[label_temp[0]]                           # eg:[2] label中的转换对应的数字
    return edge_num, state_num, violation,violation_l, label, graph, converse_graph, transition2num


# 读取正确路径（采用N次随机路径搜索）
def correct_trace_get(path_trace):
    filepath_ct = path_trace + "/correct_trace.txt"
    correct_traces = []
    with open(filepath_ct, 'r', encoding='utf-8') as trace:
        for line in trace.readlines():
            line = line.replace('[', '')
            line = line.replace(']', '')
            correct_traces.append(list(map(int, line.split(','))))
    #print("correct_traces")
    #print(correct_traces)
    return correct_traces


# 读取错误路径（采用N次随机路径搜索）
def error_trace_get(path_trace):
    filepath_ct = path_trace + "/error_trace.txt"
    error_traces = []
    with open(filepath_ct, 'r', encoding='utf-8') as trace:
        for line in trace.readlines():
            line = line.replace('[', '')
            line = line.replace(']', '')
            error_traces.append(list(map(int, line.split(','))))

    return error_traces











