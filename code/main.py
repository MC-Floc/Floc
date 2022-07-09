import tina_file as tina_file
import tina_state as tina_state
import heapq
import os
import NetConstruct as Net
import ParaFunc as ParaFunc
import Parameter as Para
import ReachabilityGraph as Graph
import TraceSearch as Trace
from Parameter import Parameter as Para
sign=0

def error():
    print("Command illegal")
    print("Try \'mfl -help\' for more information")

def is_float(s):
    s = str(s)
    if s.count('.') ==1:
        left = s.split('.')[0]
        right = s.split('.')[1]
        if right.isdigit():
            if left.count('-')==1 and left.startswith('-'):
                num = left.split['-'][-1]
                if num.isdigit():
                    return True
            elif left.isdigit():
                return True
    return False

def commandManage(a):
    s=a.split(' ')
    if s[0] == 'quit':
        global sign
        sign= 1
        return
    elif s[0] != "mfl":
        error()
        return
    elif s[1]=='-help':
        print("model fault location for ltl: mfl -z model.ktz -s model.kts -l ltl -[option]... [-v Violation Violation-l]")
        print("model fault location for deadlock: mfl -z model.ktz -[option]... [-v Violation]")
        print("quit: Exit program")
        print("Options and arguments:")
        print("mfl -para : Displays the current configuration, including various algorithms and parameters")
        print("mfl -error_path_search n : Set the path search algorithm: 0 for RTS, 1 for STS, 2 for RSTS(DEFAULT)")
        print("mfl -training_set n : Set the training set arrangement algorithm: 0 for random distribution algorithm, 1 for sorting algorithm")
        print("mfl -model n : Set learning model: 0 for BP, 1 for DNN, 2 for CNN, 3 for LR, 4 for RF(DEFAULT)")
        print("mfl -correct_path n : Set the number of correct paths, n is a positive integer (4000 for default)")
        print("mfl -error_path n : Set the number of error paths, n is a positive integer (2000 for default)")
        print("mfl -loss n : Set the loss function: 0 for MSE, 1 for Huber")
        print("mfl -optimizer n : Set the optimization function: 0 for SGD, 1 for rmsprop, 2 for Adam")
        print("mfl -lr d : Set the learning rate LR, d as a decimal")
        print("mfl -epoch n : Set the Epoch, n is a positive integer")
        print("mfl -batchsize n : Set the BatchSize, n is a positive integer")
        print("mfl -result result.txt : Set the output location of the result file, result.txt is the output location of the result file")
        print("mfl -result -c : Set the learning rate LR, d as a decimal")
        return
    elif s[1]=='-para':
        print("Current algorithms and parameters :")
        print("Error trace search method : "+str(Para.ERROR_TRACE_METHOD))
        print("Traning set arrangement method : "+str(Para.SORT))
        print("Learning model : "+str(Para.NET_METHOD))
        print("Correct trace num in training set : "+str(Para.CT_NUM))
        print("Error trace num in training set : "+str(Para.ET_NUM))
        print("Loss function : "+str(Para.LOSS_FUNCTION))
        print("Optimizer function : "+str(Para.OPTIMIZER_FUNCTION))
        print("LR : "+str(Para.LR))
        print("Epoch : "+str(Para.EPOCH))
        print("BatchSize : "+str(Para.BATCH_SIZE))
        print("Output : "+Para.Result)
        return
    elif s[1]=='-error_path_search':
        if s[2].isdigit() and int(s[2]) in [0,1,2]:
            Para.ERROR_TRACE_METHOD = int(s[2])
        else:
            error()
        return
    elif s[1]=='-training_set':
        if s[2].isdigit() and int(s[2]) in [0,1]:
            Para.SORT = int(s[2])
        else:
            error()
        return
    elif s[1]=='-model':
        if s[2].isdigit() and int(s[2]) in [0,1,2,3,4]:
            Para.NET_METHOD = int(s[2])
            ParaFunc.parameter(int(s[2]))
        else:
            error()
        return
    elif s[1]=='-error_path':
        if s[2].isdigit() and int(s[2]) > 0:
            Para.ET_NUM = int(s[2])
        else:
            error()
        return
    elif s[1]=='-correct_path':
        if s[2].isdigit() and int(s[2]) > 0:
            Para.CT_NUM = int(s[2])
        else:
            error()
        return
    elif s[1]=='-loss':
        if s[2].isdigit() and int(s[2]) in [0,1]:
            Para.LOSS_FUNCTION = int(s[2])
        else:
            error()
        return
    elif s[1]=='-optimizer':
        if s[2].isdigit() and int(s[2]) in [0,1,2]:
            Para.OPTIMIZER_FUNCTION = int(s[2])
        else:
            error()
        return
    elif s[1]=='-lr':
        if is_float(s[2]):
            Para.LR = float(s[2])
        else:
            error()
        return
    elif s[1]=='-epoch':
        if s[2].isdigit():
            Para.EPOCH = int(s[2])
        else:
            error()
        return
    elif s[1]=='-result':
        if s[2]=='-c':
            Para.Result = "None"
        else:
            try:
                filep = ""
                l=0
                for l in range(0,len(s[2])):
                    if s[2][len(s[2])-l-1] == '/':
                        l = len(s[2])-l-1
                        break
                filep = s[2][0:l]
                if not os.path.exists(filep):
                    os.makedirs(filep)
                file = open(s[2],'w')
                Para.Result=s[2]
                file.close()
            except:
                error()
        return
    elif s[1]=='-batchsize':
        if s[2].isdigit():
            Para.BATCH_SIZE = int(s[2])
        else:
            error()
        return
    elif s[1]=="-z" and s[3]=="-s" and s[5]=="-l":
        for si in range(7,len(s)):
            if s[si] != "-v":
                s[6]=s[6]+" "+s[si]
            else:
                s[7] = "-v"
                s[8] = s[si+1]
                if si+2 <len(s):
                    s[9] = s[si+2]
                else:
                    s[9] = ""
        fs=s[2].split('/')
        filepath=""
        for i in range(0,len(fs)-1):
            filepath = filepath+fs[i]+"/"
        try:
            with open(filepath+"ltl.txt",'w') as f:
                f.write(s[6])
        except:
            error()
            return
        re=tina_file.fileManage(s[2],s[4])
        if re == 0:
            return
        if len(s)>7 and s[7]=="-v":
            tina_state.copy(s[2][:-4],s[8],s[9])
        else:
            tina_state.tinaState(s[2][:-4])
        ParaFunc.random_seed()
        fs=s[2].split('/')
        filepath=""
        for i in range(0,len(fs)-1):
            filepath = filepath+fs[i]+"/"
        path = filepath[:-1]
        edge_num, state_num, violation, violation_l, label, graph, converse_graph, transition2num \
            = Graph.graph_extract(path)
        if Para.ERROR_TRACE_METHOD == 0:
            error_traces = Trace.n_random_et_ltl(converse_graph, transition2num, violation,violation_l, 5)
        elif Para.ERROR_TRACE_METHOD == 1:
            error_traces = Trace.k_shortest_error_trace(graph, violation,violation_l, 0, state_num, transition2num)
        else:
            error_traces = Trace.n_random_et_ltl(converse_graph, transition2num, violation,violation_l, 5)
            if edge_num/len(transition2num) > 500  or len(error_traces)<10:
                error_traces = Trace.k_shortest_error_trace(graph, violation,violation_l, 0, state_num, transition2num)
        print("error_traces_search: Finish!")
        correct_traces = Trace.n_random_ct_ltl(graph,transition2num,violation,violation_l,5)
        print("correct_traces_search: Finish!")
        Para.C_ALL = Para.CT_NUM # int(len(correct_traces))  # Para.CT_NUM * (int(len(violation) / 10) + 1)
        Para.E_ALL = Para.ET_NUM # int(len(error_traces))  # Para.ET_NUM * (int(len(violation) / 10) + 1)
        if (Para.NET_METHOD == 3 or Para.NET_METHOD == 4) and len(correct_traces) ==0:
            c_str = []
            for ii in range(0,len(transition2num)-1):
                c_str.append(ii)
            correct_traces.append(c_str)
        if Para.TRACE_REPEAT:
            if Para.SORT:
                correct_traces, error_traces = Trace.reconstruct_traces2(correct_traces, error_traces)
            else:
                correct_traces, error_traces = Trace.reconstruct_traces(correct_traces, error_traces)
        x, y, test = Net.matrix_construct_0(correct_traces, error_traces, len(transition2num))
        result = Net.trace_net(x, y, test, len(transition2num), 1, 1)                # [array([0.2613653], dtype=float32)]
        result_list=[]
        if Para.NET_METHOD == 3 or Para.NET_METHOD == 4:
            for index in result:
                result_list.append([index])
        else:
            for index in result:
                result_list.append(list(index))
        result = result_list
        if Para.ERROR_SET == 1:
            error_set=Trace.error_sets(error_traces,len(transition2num))
            for index in range(0,len(result)):
                if error_set[index] != 0:
                    result[index][0]=result[index][0]+1
        # 返回result中len(transition2num)个最大的结果    [(4, array([0.32418758], dtype=float32))]
        max_num_index = heapq.nlargest(len(transition2num), enumerate(result), key=lambda z: z[1])
        keys=list(transition2num.keys())
        if Para.Result == "None":
            print("Suspicion ranking list:")
            rank = 1
            for max in max_num_index:
                print(str(rank)+" : "+str(keys[max[0]]))
                rank = rank+1

        else:
            f = open(Para.Result,"w")
            f.write("Suspicion ranking list:\n")
            rank = 1
            for max in max_num_index:
                f.write(str(rank)+" : "+str(keys[max[0]])+"\n")
                rank = rank+1
        print("model fault location: Finish!")
        return
    elif s[1]=="-z":
        fs=s[2].split('/')
        filepath=""
        for i in range(0,len(fs)-1):
            filepath = filepath+fs[i]+"/"
        re=tina_file.fileManageDeadlock(s[2])
        if re == 0:
            return
        if len(s)>4 and s[3]=="-v":
            tina_state.copy(s[2][:-4],s[4],"")
        else:
            return
        ParaFunc.random_seed()
        fs=s[2].split('/')
        filepath=""
        for i in range(0,len(fs)-1):
            filepath = filepath+fs[i]+"/"
        path = filepath[:-1]
        edge_num, state_num, violation, violation_l, label, graph, converse_graph, transition2num \
            = Graph.graph_extract(path)
        if Para.ERROR_TRACE_METHOD == 0:
            error_traces = Trace.n_random_et(converse_graph, transition2num, violation, 5)
        elif Para.ERROR_TRACE_METHOD == 1:
            error_traces = Trace.k_shortest_error_trace(graph, violation,violation_l, 0, state_num, transition2num)
        else:
            error_traces = Trace.n_random_et(converse_graph, transition2num, violation, 5)
            if edge_num/len(transition2num) > 500  or len(error_traces)<10:
                error_traces = Trace.k_shortest_error_trace(graph, violation,violation_l, 0, state_num, transition2num)
        print("error_traces_search: Finish!")
        correct_traces = Trace.n_random_ct(graph,transition2num,5)
        print("correct_traces_search: Finish!")
        Para.C_ALL = Para.CT_NUM # int(len(correct_traces))  # Para.CT_NUM * (int(len(violation) / 10) + 1)
        Para.E_ALL = Para.ET_NUM # int(len(error_traces))  # Para.ET_NUM * (int(len(violation) / 10) + 1)
        if (Para.NET_METHOD == 3 or Para.NET_METHOD == 4) and len(correct_traces) ==0:
            c_str = []
            for ii in range(0,len(transition2num)-1):
                c_str.append(ii)
            correct_traces.append(c_str)
        if Para.TRACE_REPEAT:
            if Para.SORT:
                correct_traces, error_traces = Trace.reconstruct_traces2(correct_traces, error_traces)
            else:
                correct_traces, error_traces = Trace.reconstruct_traces(correct_traces, error_traces)
        x, y, test = Net.matrix_construct_0(correct_traces, error_traces, len(transition2num))
        result = Net.trace_net(x, y, test, len(transition2num), 1, 1)                # [array([0.2613653], dtype=float32)]
        result_list=[]
        if Para.NET_METHOD == 3 or Para.NET_METHOD == 4:
            for index in result:
                result_list.append([index])
        else:
            for index in result:
                result_list.append(list(index))
        result = result_list
        if Para.ERROR_SET == 1:
            error_set=Trace.error_sets(error_traces,len(transition2num))
            for index in range(0,len(result)):
                if error_set[index] != 0:
                    result[index][0]=result[index][0]+1
        # 返回result中len(transition2num)个最大的结果    [(4, array([0.32418758], dtype=float32))]
        max_num_index = heapq.nlargest(len(transition2num), enumerate(result), key=lambda z: z[1])
        keys=list(transition2num.keys())
        if Para.Result == "None":
            print("Suspicion ranking list:")
            rank = 1
            error_root = transition2num
            for max in max_num_index:
                print(str(rank)+" : "+str(keys[max[0]]))
                rank = rank+1
        else:
            f = open(Para.Result,"w")
            f.write("Suspicion ranking list:\n")
            rank = 1
            for max in max_num_index:
                f.write(str(rank)+" : "+str(keys[max[0]])+"\n")
                rank = rank+1
        print("model fault location: Finish!")
        return
    else:
        error()
        return
if __name__ == '__main__':
    path=os.getcwd()
    while 1:
        a=input(path+":")
        commandManage(a)
        if sign == 1:
            break