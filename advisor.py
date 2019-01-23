import math, random
import numpy as np
import sys, csv, os

NUM_PROBLEMS = 18
NUM_HW = 9

def get_filenumber(filename):
    switcher = {
        "bicg_output.csv": 1,
        "conv_output.csv": 2,
        "coulomb_sum_2d_output.csv": 3,
        "coulomb_sum_3d_output.csv": 4,
        "coulomb_sum_3d_iterative_output.csv": 5,
        "fourier_32_results.csv": 6,
        "fourier_50_results.csv": 7,
        "fourier_64_results.csv": 8,
        "fourier_91_results.csv": 9,
        "fourier_128_results.csv": 10,
        "fourier_197_results.csv": 11,
        "fourier_256_results.csv": 12,
        "gemm_output.csv": 13,
        "hotspot_output.csv": 14,
        "mtran_output.csv": 15,
        "nbody_output.csv": 16,
        "reduction_output.csv": 17,
        "sort_output.csv": 18,
        "sort_result.csv": 18
    }
    return switcher.get(filename, 19)

def get_dirnumber(dirname):
    switcher = {
        "CPU-dual-E5-2650": 1,
        "GPU-P100": 2,
        "GPU-K20": 3,
        "GPU-Vega56": 4,
        "GPU-1070": 5,
        "GPU-680": 6,
        "GPU-TitanV": 7,
        "MIC-5110P": 8,
        "GPU-750": 9,
    }
    return switcher.get(dirname, 10)

def count_best(lst,percent):
    count = 0.0
    m = min(lst)
    for i in range(len(lst)):
        if (lst[i] <= m+m*percent/100):
            count += 1
    return count*100/len(lst)

def square(list):
    return [i ** 2 for i in list]

def meandev(predictions, targets):
    diff = list(np.array(predictions)/np.array(targets))
    for i in range(len(diff)):
        if (diff[i] < 1) and (diff[i] != 0):
            diff[i] = 1/diff[i]
    return np.mean(diff)

def rmsd(predictions, targets):
    diff = list(np.array(predictions)/np.array(targets))
    for i in range(len(diff)):
        if (diff[i] < 1) and (diff[i] != 0):
            diff[i] = 1/diff[i]

    return np.sqrt(np.mean(square(diff)))

def get_prediction(data,architecture,problem):
    #0 - CPU-dual-ES-2650
    #1 - GPU-P100
    #2 - GPU-K20
    #3 - GPU-Vega56
    #4 - GPU-1070
    #5 - GPU-680
    #6 - GPU-TitanV
    #7 - MIC-5110P
    #8 - GPU-750
    
    order = [1,2,3,4,5,6,7,8]

    if architecture == 0:
        order = [7,3,2,1,5,8,4,6]
    if architecture == 1:
        order = [2,5,8,4,6,3,7,0]
    if architecture == 2:
        order = [1,5,8,4,6,3,7,0]
    if architecture == 3:
        order = [4,8,5,6,1,2,7,0]
    if architecture == 4:
        order = [8,5,6,1,2,3,7,0]
    if architecture == 5:
        order = [8,1,2,4,6,3,7,0]
    if architecture == 6:
        order = [4,8,5,1,2,3,7,0]
    if architecture == 7:
        order = [0,3,4,8,5,6,1,2]
    if architecture == 8:
        order = [5,1,2,4,6,3,7,0]

    closest_architecture = 0

    for i in range(len(order)):
        if data[order[i]][problem] != -1:
            closest_architecture = order[i]
            break;
    return data[closest_architecture][problem]

lines = sys.stdin.readlines()

directory = ""
dirorder = 1
flag = 0
values = [-1.0] * NUM_PROBLEMS
allvalues = []
emptyvalues = 1

###
accepted_slowdown = 20
probability = 80
###

for line in lines:
    data = []

    newdirectory = os.path.dirname(os.path.abspath(line)).split("/")[-1]
    if (flag == 0):
        directory = newdirectory
        while (dirorder != get_dirnumber(newdirectory)):
            emptyline = [-1.0] * NUM_PROBLEMS
            allvalues.append(emptyline)
            dirorder += 1
        flag = 1

    if (newdirectory != directory) and (emptyvalues == 0):
        newvalues = []
        for i in range(len(values)):
            if values[i] >= 0:
                newvalues.append(values[i])
            else: 
                newvalues.append(-1.0)
            values[i] = -1.0

        while (dirorder != get_dirnumber(newdirectory)):
            emptyline = [-1.0] * NUM_PROBLEMS
            allvalues.append(emptyline)
            dirorder += 1
        allvalues.append(newvalues)
        directory = newdirectory
        dirorder += 1
        emptyvalues = 1

    csv_file = line[:-1]

    filename = csv_file.split("/")[-1]

    with open(csv_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            reader.next()
            for row in reader:
                if len(row) < 2:
                    break
                time = float(row[1])
                data.append(time)

    percent_best = count_best(data,int(accepted_slowdown))

    inv_probability = 1.0-(float(probability)/100)
    inv_best = 1.0-(percent_best/100)

    if probability >= 100:
        num_steps = float(math.ceil(inv_best * len(data) + 1))
    else:
        num_steps = float(math.ceil(math.log(inv_probability,inv_best)))

    if len(data) > 0:
        number = get_filenumber(filename)-1
        if number < len(values):
            values[get_filenumber(filename)-1] = num_steps/len(data)
            emptyvalues = 0

if (emptyvalues == 0):
    newvalues = []
    for i in range(len(values)):
        if values[i] >= 0:
            newvalues.append(values[i])
        else:
            newvalues.append(-1.0)
        values[i] = -1.0
    allvalues.append(newvalues)

while len(allvalues) < NUM_HW:
    emptyline = [-1.0] * NUM_PROBLEMS
    allvalues.append(emptyline)

predictions = []

for i in range(len(allvalues)):
    newpredictions = []
    for j in range(len(allvalues[i])):
        if allvalues[i][j] == -1.0:
            newpredictions.append(-1.0)
        else:
            newpredictions.append(get_prediction(allvalues,i,j))
    predictions.append(newpredictions)

list1 = []
list2 = []
for i in range(len(allvalues)):
    for j in range(len(allvalues[i])):
        if (allvalues[i][j] != -1.0):
            list1.append(allvalues[i][j])
            list2.append(predictions[i][j])

print "PREDICTION RMSD: " + str(rmsd(list1,list2))
print "================================================"

for i in range(100):
    dumblist = []
    for j in range(len(list1)):
        dumblist.append(float(i)/100+0.01)
    print str(float(i)/100+0.01) + ": " + str(rmsd(list1,dumblist))

print "\n\n\n\n\n\n"

print "PREDICTION AVERAGE RELATIVE DEVIATION: " + str(meandev(list1,list2))
print "================================================"

for i in range(100):
    dumblist = []
    for j in range(len(list1)):
        dumblist.append(float(i)/100+0.01)
    print str(float(i)/100+0.01) + ": " + str(meandev(list1,dumblist))
