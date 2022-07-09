# -*- coding: utf-8 -*-
import heapq
import random
import time

import ParaFunc
from Parameter import Parameter as Para
from ReachabilityGraph import graph_extract

global_error_traces = []
temp_error_traces = []
temp_error_places = []
global_error_places = []


def reconstruct_traces(correct_traces, error_traces):
    correct_traces = [list(t) for t in set(tuple(_) for _ in correct_traces)]
    error_traces = [list(t) for t in set(tuple(_) for _ in error_traces)]
    # 如果路径数量不够，直接复制多个
    while len(correct_traces) < Para.C_ALL:
        if len(correct_traces) == 0:
            break
        correct_traces.extend(correct_traces)
    while len(error_traces) < Para.E_ALL:
        error_traces.extend(error_traces)
    random.shuffle(correct_traces)  # 将元素随机排列
    random.shuffle(error_traces)
    while len(correct_traces) != Para.C_ALL:
        if len(correct_traces) == 0:
            break
        correct_traces.pop()
    while len(error_traces) != Para.E_ALL:
        error_traces.pop()

    return correct_traces, error_traces

def find_same(temp,traces):
    max = 1
    maxi = []
    for i in traces:
        cont = 0
        for j in temp:
            if j in i:
                cont = cont+1
        if max < cont:
            max = cont
            maxi.clear()
            maxi.append(i)
        elif max == cont:
            maxi.append(i)
        elif max > len(i):
            break
    return maxi

def reconstruct_traces2(correct_traces, error_traces):
    correct_traces = [list(t) for t in set(tuple(_) for _ in correct_traces)]
    error_traces = [list(t) for t in set(tuple(_) for _ in error_traces)]
    correct_traces.sort(key=len,reverse = True)
    error_traces.sort(key=len,reverse = True)
    correct_traces_sort = []
    error_traces_sort = []
    while len(correct_traces) != 0:
        temp = correct_traces[0]
        correct_traces_sort.append(temp)
        correct_traces.remove(temp)
        sameset = find_same(temp,correct_traces)
        sameset.sort(key=len,reverse = True)
        for i in sameset:
            correct_traces_sort.append(i)
            correct_traces.remove(i)

    while len(error_traces) != 0:
        temp = error_traces[0]
        error_traces_sort.append(temp)
        error_traces.remove(temp)
        sameset = find_same(temp,error_traces)
        sameset.sort(key=len,reverse = True)
        for i in sameset:
            error_traces_sort.append(i)
            error_traces.remove(i)
    if len(correct_traces_sort) !=0:
        c_num = (int)(Para.C_ALL/len(correct_traces_sort))
        c_ret = Para.C_ALL-c_num*len(correct_traces_sort)
    e_num = (int)(Para.E_ALL/len(error_traces_sort))
    e_ret = Para.E_ALL-e_num*len(error_traces_sort)
    ct = []
    et = []
    if len(correct_traces_sort) > Para.C_ALL:
        ct = correct_traces_sort[0:Para.C_ALL]
    else:
        i=0
        for tr in correct_traces_sort:
            i=i+1
            for j in range(0,c_num):
                ct.append(tr)
            if i<=c_ret:
                ct.append(tr)

    if len(error_traces_sort) > Para.E_ALL:
        et = error_traces_sort[0:Para.E_ALL]
    else:
        i=0
        for tr in error_traces_sort:
            i=i+1
            for j in range(0,e_num):
                et.append(tr)
            if i <= e_ret:
                et.append(tr)
    return ct,et

def retrace(correct_traces, error_traces, length):
    correct_traces_len = []
    for trace in correct_traces:
        vertex = [0] * (length+1)
        for tr in trace:
            vertex[tr] += 1
        _len = 0
        for i in vertex:
            if i != 0:
                _len = _len+1
        vertex[-1] = _len
        correct_traces_len.append(vertex)

    error_traces_len = []
    for trace in error_traces:
        vertex = [0] * (length+1)
        for tr in trace:
            vertex[tr] += 1
        _len = 0
        for i in vertex:
            if i != 0:
                _len = _len+1
        vertex[-1] = _len
        error_traces_len.append(vertex)
    return correct_traces_len,error_traces_len

def get_key(x):
    return int(x[-1])

def find_same2(temp,traces):
    max = 1
    maxi = []
    for i in traces:
        cont = 0
        for j in range(0,len(temp)-1):
            if temp[j] == i[j]:
                cont = cont+1
        if max < cont:
            max = cont
            maxi.clear()
            maxi.append(i)
        elif max == cont:
            maxi.append(i)
        elif max > i[-1]:
            break
    return maxi

def reconstruct_traces3(correct_traces, error_traces, length):
    correct_traces = [list(t) for t in set(tuple(_) for _ in correct_traces)]
    error_traces = [list(t) for t in set(tuple(_) for _ in error_traces)]
    correct_traces, error_traces=retrace(correct_traces, error_traces, length)
    correct_traces.sort(key=get_key,reverse = True)
    error_traces.sort(key=get_key,reverse = True)
    correct_traces_sort = []
    error_traces_sort = []

    while len(correct_traces) != 0:
        temp = correct_traces[0]
        correct_traces_sort.append(temp[:-1])
        correct_traces.remove(temp)
        sameset = find_same2(temp,correct_traces)
        sameset.sort(key=get_key,reverse = True)
        for i in sameset:
            correct_traces_sort.append(i[:-1])
            correct_traces.remove(i)

    while len(error_traces) != 0:
        temp = max(error_traces, key=len)
        error_traces_sort.append(temp[:-1])
        error_traces.remove(temp)
        sameset = find_same2(temp,error_traces)
        sameset.sort(key=len,reverse = True)
        for i in sameset:
            error_traces_sort.append(i[:-1])
            error_traces.remove(i)

    if len(correct_traces_sort) !=0:
        c_num = (int)(Para.C_ALL/len(correct_traces_sort))
        c_ret = Para.C_ALL-c_num*len(correct_traces_sort)
    e_num = (int)(Para.E_ALL/len(error_traces_sort))
    e_ret = Para.E_ALL-e_num*len(error_traces_sort)
    ct = []
    et = []
    if len(correct_traces_sort) > Para.C_ALL:
        ct = correct_traces_sort[0:Para.C_ALL]
    else:
        i=0
        for tr in correct_traces_sort:
            i=i+1
            for j in range(0,c_num):
                ct.append(tr)
            if i<=c_ret:
                ct.append(tr)

    if len(error_traces_sort) > Para.E_ALL:
        et = error_traces_sort[0:Para.E_ALL]
    else:
        i=0
        for tr in error_traces_sort:
            i=i+1
            for j in range(0,e_num):
                et.append(tr)
            if i <= e_ret:
                et.append(tr)
    return ct,et

def error_sets(error_traces,length):
    error_set=[0]*length
    if Para.MATRIX_CONSTRUCT_METHOD == 1 and Para.SORT == 1:
        for i in error_traces:
            for j in range(0,len(i)):
                if i[j] !=0:
                    error_set[j]=1
    else:
        for i in error_traces:
            for j in i:
                error_set[j]=1
    return error_set

def random_ct_ltl(graph, transition2num, violation, violation_l):
    correct_traces = []
    path_state = []
    path_transition = []
    violation=violation+violation_l
    def get_trace():
        node = random.choice(graph[0])                                  # 随机选择节点
        while node[0] not in path_state:                                # 当没有循环
            if node[0] not in violation:                                # 如果不是违例状态
                path_state.append(node[0])                              # 加入
                path_transition.append(transition2num[node[1]])
                if not graph[node[0]]:                                  # 如果正常结束，则退出
                    break
                node = random.choice(graph[node[0]])                    # 随机选择下一个
            else:                                                       # 如果是违例状态
                path_state.clear()                                      # 全都清空
                path_transition.clear()
                break

    starttime = time.time()
    while len(correct_traces) < Para.C_ALL:
        get_trace()
        if path_transition:
            if Para.REMOVE_DUPLICATE:
                correct_traces.append(list(set(path_transition.copy())))
            else:
                correct_traces.append(path_transition.copy())
        path_state = []
        path_transition = []
        endtime = time.time()
        t = endtime - starttime
        if t > 3:
            break
    return correct_traces

def n_random_ct_ltl(graph, transition2num, violation, violation_l, N):
    correct_traces = []
    for i in range(0, N):
        correct_traces += random_ct_ltl(graph, transition2num,violation,violation_l)           # 做n次随机搜索
        correct_traces = [list(t) for t in set(tuple(_) for _ in correct_traces)]  # 去重
    return correct_traces

num = 0
def random_et_ltl(graph, violation, violation_l, transition2num):    # graph是翻转图
    error_traces = []
    path_state = []
    path_transition = []

    def get_trace(vs):              # 从违例状态倒着找
        path_state.clear()
        path_transition.clear()
        node = random.choice(graph[vs])
        while 1:
            path_state.append(node[0])
            path_transition.append(transition2num[node[1]])
            if node[0] == 0:
                break;
            if node[0] == vs:
                num=1
            node = random.choice(graph[node[0]])

    temp = int(Para.E_ALL / (len(violation)+len(violation_l)) )             # 每个违例状态的平均数量

    for v in violation:                                 # 为每个违例状态搜索平均数量的错误路径
        starttime = time.time()
        for i in range(temp):
            get_trace(v)
            if Para.REMOVE_DUPLICATE:
                error_traces.append(list(set(path_transition.copy())))
            else:
                error_traces.append(path_transition.copy())
            endtime = time.time()
            t = endtime - starttime
            if t > 5:
                break

    for v in violation_l:                                 # 为每个违例状态搜索平均数量的错误路径
        starttime = time.time()
        for i in range(temp):
            get_trace(v)
            if num == 1:
                if Para.REMOVE_DUPLICATE:
                    error_traces.append(list(set(path_transition.copy())))
                else:
                    error_traces.append(path_transition.copy())
            endtime = time.time()
            t = endtime - starttime
            if t > 5:
                break
    starttime = time.time()
    violation = violation+violation_l
    while len(error_traces) < Para.E_ALL:     # 随机选择一个违例状态，补充到规定的错误路径数量
        v = random.choice(violation)
        get_trace(v)
        if Para.REMOVE_DUPLICATE:
            error_traces.append(list(set(path_transition.copy())))
        else:
            error_traces.append(path_transition.copy())

        endtime = time.time()
        t = endtime - starttime
        if t > 10:
            break
    return error_traces

def n_random_et_ltl(graph, transition2num, violation, violation_l, N):
    error_traces = []
    for i in range(0, N):
        error_traces += random_et_ltl(graph, violation, violation_l, transition2num)
        error_traces = [list(t) for t in set(tuple(_) for _ in error_traces)]
    return error_traces

def k_shortest_error_trace(graph, violation, violation_l, start, num, transition2num):
    dist = {}
    visited = {}
    # prev_points[k]记录了到点k的最短路径中的前驱节点
    prev_points = {}
    prev_transitions = {}
    error_traces = []
    for i in range(0, num):
        dist[i] = 0x3f3f3f3f
        visited[i] = False
        prev_points[i] = []
        prev_transitions[i] = []
    dist[start] = 0
    heap = [(0, start)]
    heapq.heapify(heap)
    while heap:
        distance, node = heapq.heappop(heap)
        if visited[node]:
            continue
        visited[node] = True
        for next_node in graph[node]:
            if not visited[next_node[0]]:
                if dist[next_node[0]] > dist[node] + 1:
                    dist[next_node[0]] = dist[node] + 1
                    prev_points[next_node[0]].clear()
                    prev_points[next_node[0]].append(node)
                    prev_transitions[next_node[0]].clear()
                    prev_transitions[next_node[0]].append(next_node[1])
                    heapq.heappush(heap, (dist[next_node[0]], next_node[0]))
                elif dist[next_node[0]] == dist[node] + 1:
                    prev_points[next_node[0]].append(node)
                    prev_transitions[next_node[0]].append(next_node[1])
                    heapq.heappush(heap, (dist[next_node[0]], next_node[0]))
    violation=violation+violation_l
    for v in violation:
        global_error_traces.clear()
        get_paths(start, v, prev_transitions, prev_points, transition2num)
        for i in range(0, len(global_error_traces)):
            error_traces.append(global_error_traces[i].copy())
        error_traces=[list(t) for t in set(tuple(_) for _ in error_traces)]

    return error_traces

def get_paths(start, end, prev_transitions, prev_points, transition2num):
    if start == end:
        temp_error_traces.reverse()
        temp_error_places.reverse()
        if Para.REMOVE_DUPLICATE:
            global_error_traces.append(list(set(temp_error_traces.copy())))
            global_error_places.append(set(temp_error_places.copy()))
        else:
            global_error_traces.append(temp_error_traces.copy())
            global_error_places.append(temp_error_places.copy())
        temp_error_traces.reverse()
        temp_error_places.reverse()
    else:
        for i in range(0, len(prev_points[end])):
            temp_error_traces.append(transition2num[prev_transitions[end][i]])
            temp_error_places.append(prev_points[end][i])
            if len(global_error_traces)>1000:
                pass
            else:
                get_paths(start, prev_points[end][i], prev_transitions, prev_points, transition2num)
            temp_error_traces.pop(-1)
            temp_error_places.pop(-1)

def random_ct(graph, transition2num):
    correct_traces = []
    path_state = []
    path_transition = []

    def get_trace():
        node = random.choice(graph[0])
        while node[0] not in path_state:
            path_state.append(node[0])
            path_transition.append(transition2num[node[1]])
            if graph[node[0]]:
                node = random.choice(graph[node[0]])
            else:
                path_state.clear()
                path_transition.clear()
                break

    starttime = time.time()
    while len(correct_traces) < Para.C_ALL:           #修改为小于C_temp
        get_trace()
        if path_transition:
            if Para.REMOVE_DUPLICATE:
                correct_traces.append(list(set(path_transition.copy())))
            else:
                correct_traces.append(path_transition.copy())
        endtime = time.time()
        t = endtime - starttime
        if t > 10:
            break

    return correct_traces

def random_et(graph, violation, transition2num):    # graph是翻转图
    error_traces = []
    path_state = []
    path_transition = []

    if 0 in graph:                                # 因为没有状态会到达初始状态，所以把0删掉
        del graph[0]

    def get_trace(vs):
        path_state.clear()
        path_transition.clear()
        node = random.choice(graph[vs])
        while node[0] != 0:
            path_state.append(node[0])
            path_transition.append(transition2num[node[1]])
            node = random.choice(graph[node[0]])

    temp = int(Para.E_ALL / len(violation))             # 每个违例状态的平均数量
    for v in violation:                                 # 为每个违例状态搜索平均数量的错误路径
        starttime = time.time()
        for i in range(temp):
            get_trace(v)
            if Para.REMOVE_DUPLICATE:
                error_traces.append(list(set(path_transition.copy())))
            else:
                error_traces.append(path_transition.copy())
            endtime = time.time()
            t = endtime - starttime
            if t > 5:
                break
    starttime = time.time()
    while len(error_traces) < Para.E_ALL:     # 随机选择一个违例状态，补充到规定的错误路径数量
        v = random.choice(violation)
        get_trace(v)
        if Para.REMOVE_DUPLICATE:
            error_traces.append(list(set(path_transition.copy())))
        else:
            error_traces.append(path_transition.copy())

        endtime = time.time()
        t = endtime - starttime
        if t > 10:
            break

        # state_num_dict = dict(Counter(path_state))
        # repeat_node = [key for key, value in state_num_dict.items()if value > 1]
        # for n in repeat_node:
        #     index = [i for i, x in enumerate(path_state) if x == n]
        #     if index:
        #         del path_state[index[0]:index[-1]]
        #         del path_transition[index[0]:index[-1]]

    return error_traces

def n_random_ct(graph, transition2num, N):
    correct_traces = []
    for i in range(0, N):
        correct_traces += random_ct(graph, transition2num)           # 做n次随机搜索
        correct_traces = [list(t) for t in set(tuple(_) for _ in correct_traces)]  # 去重

    return correct_traces

def n_random_et(graph, transition2num, violation, N):
    error_traces = []
    for i in range(0, N):
        error_traces += random_et(graph, violation, transition2num)
        error_traces = [list(t) for t in set(tuple(_) for _ in error_traces)]

    return error_traces
