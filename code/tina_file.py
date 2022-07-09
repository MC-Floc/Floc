import os
def fileManage(ktz,kts):
    try:
        filename=ktz[:-4]
        if ktz[:-4] != kts[:-4]:
            with open(kts,'r') as f:
                line=f.readlines()
            with open(filename+".kts",'w') as f:
                for l in line:
                    f.write(l)
        cmd="ktzio "+filename+".ktz "+filename+".aut"
        os.system(cmd)
        with open(filename+".aut",'r') as f:
            line=f.readlines()
        with open(filename+".aut",'w') as f:
            for l in line:
                if l[0] != '#':
                    f.write(l)
    except:
        print("Command illegal")
        print("Try \'mfl -help\' for more information")
        return 0
    return 1
def fileManageDeadlock(ktz):
    try:
        filename=ktz[:-4]
        cmd="ktzio "+filename+".ktz "+filename+".aut"
        os.system(cmd)
        with open(filename+".aut",'r') as f:
            line=f.readlines()
        with open(filename+".aut",'w') as f:
            for l in line:
                if l[0] != '#':
                    f.write(l)
    except:
        print("Command illegal")
        print("Try \'mfl -help\' for more information")
        return 0
    return 1