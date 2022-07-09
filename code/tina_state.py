import os
import shutil


def tinaState(file):
    outname = 1             # 反复加入路径使用
    left = "[]("
    mid = " => ()"
    right = ")"
    s_and = " /\\ "
    s_or = " \/ "
    selt = "selt "            # LTL检查命令使用
    fs=file.split('/')
    filepath=""
    for i in range(0,len(fs)-1):
        filepath = filepath+fs[i]+"/"
    filename = fs[len(fs)-1]+".ktz -f \""
    with open(filepath+"ltl.txt",'r') as f_ltl:
        ltl = f_ltl.readline()
    ltl_p = "\" -p "
    out = "path/"+str(outname)+".txt"
    if not os.path.exists(filepath+"path"):
        os.mkdir(filepath+"path")
    cmd = selt+filepath+filename+ltl+ltl_p+filepath+out # selt test/1.ktz -f “[](customer_started => <>Rejected)” -p 1.txt
    os.system(cmd)

    violations = []
    violations_l = []
    while 1:                  # 一直循环到TRUE为止
        outname = outname+1
        lines=""
        vio=""
        with open(filepath+out,'r') as f:
            lines = f.readlines()
            transitions = []
            for l in lines:
                if l[0] == '*':
                    vio=l
                    break
        if lines[3] == "TRUE\n":
            break
        # 处理违例状态
        i=0
        sign=0
        for i in range(0,len(vio)):
            if vio[i] == ':':
                i=i+2
                break
        if vio[i:i+6] == 'L.dead':
            sign=1
            i=i+7
        vio = vio[i:]                  # 路径中的违例状态
        state2vio = []
        cont = -1
        with open(filepath+fs[len(fs)-1]+".kts",'r') as f:   # kts中的状态对应
            lines = f.readlines()
            for l in lines:
                if l[0:5] == "props":
                    cont=cont+1
                    if l[6:] == vio:
                        if sign == 1:
                            violations.append(cont)
                        else:
                            violations_l.append(cont)
        violations = list(set(violations))         #去重
        violations_l = list(set(violations_l ))
        out = "path/"+str(outname)+".txt"
        violist = vio[:-1].split(' ')
        all=violist[0]
        for vi in range(1,len(violist)):
            all=all+s_and+violist[vi]
        ltl = ltl+s_or+"<>("+all+")"
        cmd = selt+filepath+filename+ltl+ltl_p+filepath+out
        if len(cmd)>450:
            break
        os.system(cmd)
    with open(filepath+"vs.txt",'w') as f:
        for v in violations:
            f.write(str(v))
            f.write(" ")
    with open(filepath+"vs_l.txt",'w') as f:
        for v in violations_l:
            f.write(str(v))
            f.write(" ")
    with open(filepath+fs[len(fs)-1]+".aut",'r') as f1:
        lines=f1.readlines()
        lines.pop(-1)
        with open(filepath+"rg.txt",'w') as f2:
            for l in lines:
                f2.write(l)
def copy(file,v,vl):
    fs=file.split('/')
    filepath=""
    for i in range(0,len(fs)-1):
        filepath = filepath+fs[i]+"/"
    if v!=filepath+"vs.txt":
        shutil.copyfile(v,filepath+"vs.txt")
    if vl=="" or vl==None:
        if vl!=filepath+"vs_l.txt":
            with open(filepath+"vs_l.txt",'w') as f:
                f.close()
    else:
        shutil.copyfile(vl,filepath+"vs_l.txt")
    with open(filepath+fs[len(fs)-1]+".aut",'r') as f1:
        lines=f1.readlines()
        lines.pop(-1)
        with open(filepath+"rg.txt",'w') as f2:
            for l in lines:
                f2.write(l)