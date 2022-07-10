# MC-Floc Instructions

## 1、Data Instructions

​	All data sets are stored in the "data" folder. Folder 1-7 are LTL data sets and Folder 8-10 are deadlock data sets. 

The behavior of these models are explained hereafter: 

M1: Processes sharing resource. 

M2: TCPcondis{https://mcc.lip6.fr/pdf/TCPcondis-form.pdf}. 

M3: smart contract.

M4: ShieldRVs{https://mcc.lip6.fr/pdf/ShieldRVs-form.pdf}

M5: RobotManipulation{https://mcc.lip6.fr/pdf/RobotManipulation-form.pdf}

M6: ECMA+{http://daze.ho.ua/tinaz.zip}

M7: PAM {https://www.cristal.univ-lille.fr/iwaise12/tutorials/rdp-1-ptn.pdf}

![1](picture\1.png)

### 1.1 LTL Data sets

​	Folders 1-4 are responsiveness data sets, and each folder includes 15 cases. Folders 5-7 are until release and imply data sets, and each folder includes 6 cases. The files are shown in the following figure. 

* ndr file is the ndr representation of Petri net. 

* label.txt is the root cause of the error. 

* ltl.txt is a violation of the LTL nature. 
* rg.txt is the state reachable graph
* vs.txt is acyclic violation status
* vs_l.txt is cycle violation status

![2](picture\2.png)

### 1.2 Deadlock Data sets

​	Folders 8-10 are deadlock data sets, folder 8  includes 30 cases，folder 9 and 10 include 15 cases. The files are shown in the following figure. 

* ktz file is the ktz representation of Petri net. 

* label.txt is the root cause of the error. 

* rg.txt is the state reachable graph
* vs.txt is violation status

![3](picture\3.png)

## 2、Code Instructions

​	All code files are stored in the "code" folder. 

* ReachabilityGraph.py: extract various information of the state reachable graph.
* tina_ state.py: search for violation status.
* tina_file.py: transform Petri net models into various forms.
* TraceSearch. py: path search.
* main.py: fault location file, fault location according to the selected model.
* NetClass.py: establish neural network files and compile the structure of various networks.
* NetConstruct.py: describes the algorithm based on coverage.
* ParaFunc.py: defines the default parameters of several networks, which can be modified.
* Parameter.py: all parameters of the whole experiment are defined and can be modified.
* main.spec: package the file and use pyinstaller to package it into a command line file.

## 3、Configuration Instructions

​	This tool is a command line tool running on Windows system. Click main.exe can be run directly. Before running the program, you first need to download and install [Tina]([The TINA toolbox Home Page - TIme petri Net Analyzer - by LAAS/CNRS](https://projects.laas.fr/tina/papers.php)), which can check the model.

## 4、Tool Instructions

​	Input "mfl -help" to view the tool help, and the output is as follows.

* The first is the model checking fault location.
  * use "mfl -z model.ktz -s model.kts -l ltl -[option]... [-v Violation Violation-l]" to locating LTL error. KTS and KTZ are two representations of Petri nets, which can be converted by Tina tool. LTL is the detected LTL property. In addition, the violation status file can be directly input through -v. Currently, until and release can only be located by entering the violation status file.
  * use "mfl -z model.ktz -v Violation" to locating deadlock error. KTZ is a representation of Petri nets, which can be converted by Tina tool. You can enter the violation status file directly through -v. At present, it can only be located by entering the violation status file.

* Other functions can set various parameters, including the path search algorithm used, the training set arrangement algorithm, the learning model, the number of correct and wrong paths, the result output location, and so on. The specific commands and explanations can be viewed in the tool or the following figure.

![4](picture\4.png)
