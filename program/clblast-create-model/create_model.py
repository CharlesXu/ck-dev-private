#!/usr/bin/python

# Author : Marco Cianfriglia



################################################################################
################################################################################

# IMPORTS
################################################################################

import ck.kernel as ck
import re
import argparse
import os
from shutil import copyfile
import json
from sklearn import tree
import copy
import pydotplus 
import graphviz
import random
import sys
import glob
import math






################################################################################
################################################################################

# GLOBAL VARIABLES
################################################################################

# Ratio between test and training sets
DEFAULT_RATIO=80
pipeline_output='con' 
program = 'clblast-tune-ml'
program_check = 'clblast-check-ml'
cmd_key = 'xgemm-fp32'
platform = 'nvidia-dgx'
run=3
num_leaf = 0
output_dir =''
json_out_dir = ''

header = """
#ifndef DVDT_INFER_H
#define DVDT_INFER_H

#include <vector>

#include "clblast.h"

namespace clblast{

struct dvdtKernelInfo{
        std::vector<std::string> routines_vett;
        std::string k_name;
    };
    
template <typename T> 
        struct dvdtKernelInfo GetConf(const Layout layout, const Transpose a_transpose,const Transpose b_transpose,
        const size_t m, const size_t n, const size_t k,
        const T alpha, const size_t a_offset, const size_t a_ld,
        const size_t b_offset, const size_t b_ld,
        const T beta, const size_t c_offset, const size_t c_ld, int *flag){
        struct dvdtKernelInfo k_info;

        std::vector<std::string> routines_vett = {"Copy","Pad","Transpose","Padtranspose","KernelSelection"};
        std::string k_name = "";

        """


close_fun = """

k_info.routines_vett = routines_vett;
k_info.k_name = k_name;
return k_info;
}
"""


footer = """ 

template struct dvdtKernelInfo GetConf<float>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const float alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const float beta, const size_t c_offset, const size_t c_ld, int *flag);

template struct dvdtKernelInfo GetConf<double>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const double alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const double beta, const size_t c_offset, const size_t c_ld, int *flag);

template struct dvdtKernelInfo GetConf<float2>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const float2 alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const float2 beta, const size_t c_offset, const size_t c_ld, int *flag);

template struct dvdtKernelInfo GetConf<double2>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const double2 alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const double2 beta, const size_t c_offset, const size_t c_l, int *flag);


}
#endif
"""


################################################################################
################################################################################

# GET TRAINING DATASET
################################################################################

def getFeatureNames():
    feature_names = ['m', 'n', 'k', 'm * n * k', 'kernel_name']
    return feature_names

def getKernelId(kernels_array, kernel_name):
  
    for i in range(len(kernels_array)):
        if kernel_name == kernels_array[i]:
            return i

def getTrainingFromDirectory(kernels_array,output_dir):
    X=[] #features
    Y=[] #labels
    Z=[] #extra features (not used now)
   
    count=0
    for e in os.listdir(output_dir):
        if not  os.path.isfile(output_dir + os.sep + e) :
            continue

        if 'tune' in e:
            continue
        
        json_data = open(output_dir + os.sep + e)
        d = json.load(json_data)
               
        m = int(d['arg_m'])
        n = int(d['arg_n'])
        k = int(d['arg_k'])
       
        if len(d['results']) <= 0 :
            print ("[WARN] : The tuner was not able to find any valid configuration")
            print ("[WARN] : Skipping the matrix")
            json_data.close()
            continue

        kernel = d['kernel_family']
        kernel = kernel[:-2]
        # print "KERNEL : " + kernel
        precision = d['precision']
        device = d['device']
        device_type = d['device_type']
        device_vendor = d['device_vendor']
        device_core_clock = d['device_core_clock']
        device_compute_units = d['device_compute_units']
        json_data.close()

        tmp_file = ( 'tmp-ck-clblast-tune-'+kernel+'-multiconf-'+str(m)+ '-'+str(n)+ '-'+str(k)+ '-'+precision + '.json')
        json_data = open(output_dir + os.sep + tmp_file)
        d = json.load(json_data)
        

        # Could be interesting to verify the configuration by using the client 
        if 'GFLOPS' in d['statistics']['best_configuration'] : 
            gflops = d['statistics']['best_configuration']['GFLOPS']
            parameters = d['statistics']['best_configuration']['parameters']
            time = d['statistics']['best_configuration']['time']
        else:
            gflops = 0
            parameters = None
            time = sys.float_info.max

        
        kernel_id = getKernelId(kernels_array, kernel)
        conf_sign = genConfSign(parameters)
      	
        count +=1
        # NOTE : For now we use only m, n, k as features and gflops as labels
        X.append([ m,n,k])
        #X.append({'m': m, 'n' : n , 'k' : k})
        Y.append(conf_sign)
        #Y.append(str(math.ceil(gflops)))
        Z.append({
        	'm': m, 
        	'n' : n , 
        	'k' : k, 
        	'kernel': kernel,
        	'kernel_id' : kernel_id,
		'conf_sign' : conf_sign,
        	'parameters': parameters, 
        	'time': time, 
        	'precision' : precision,
        	'device': device,
        	'device_type' : device_type,
        	'device_vendor' : device_vendor,
        	'device_core_clock': device_core_clock,
        	'device_compute_units': device_compute_units,
        	'file_path' : e,
		'gflops' : gflops,
        	})
        json_data.close()

    T={'X' : X, 'Y' : Y, 'Z' : Z}
    return T


################################################################################
################################################################################

# DUMP TRAINING DATASET
################################################################################

def dumpTrainingToFile(training_set, output_file_path='/tmp/out.json'):
    out_file = open(output_file_path,'w')
    json.dump(training_set,out_file, sort_keys = True,indent = 4)
    out_file.close()


def updateKernelOnJson(training_set, exp_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(training_set['Z'])):
        if training_set['W'][i]['used'] == 1 :
            e = training_set['Z'][i]
            f_in = open(exp_dir + os.sep + e['file_path'])
            json_data = json.load(f_in)

            
            json_data['kernel_family'] = training_set['W'][i]['kernel']

            f_out = open(out_dir + os.sep + e['file_path'],'w')
            json.dump(json_data,f_out, sort_keys = True, indent = 4)
            f_in.close()
            f_out.close()


# Create the Training Set
def createTrainingSet(arg):
    
    DATASET=[]
    if arg.dataset_dir != None:
        copyDataset(arg.dataset_dir, output_dir)
    DATASET = getTrainingFromDirectory(arg.kernel_name,output_dir)
    signSet = genConfSet(arg.kernel_name, DATASET['Z'])
    routines_names = generateRoutinesNamesEnhanced(arg.kernel_name,DATASET['Z'], signSet)

    DATASET['W'] = routines_names
    
    #print DATASET['W']
    ratio = DEFAULT_RATIO
    if myarg.ratio != None:
        ratio = int(myarg.ratio)
    
    DATASET = splitDataset(DATASET, ratio)
    return DATASET


################################################################################
################################################################################

# DECISION TREE
################################################################################



# BUILD A DECISION TREE
# (Using the training dataset)
def createDecisionTree(training_set,depth=None, tree_min_samples_leaf=1):
    clf = tree.DecisionTreeClassifier(max_depth = depth, min_samples_leaf = tree_min_samples_leaf)
    clf = clf.fit(training_set['X'],training_set['Y'])
    return clf


def kernelToFamily(kernel):
    k=re.sub(r'_\d+', '', kernel)
    k = k.split('_')
    f=''
    for i in k:
        f += i
    return f.capitalize()

def generateRoutinesNamesEnhanced(kernels_name, training_set, signSet):
    routines_names = []
    count_vett=[]
    
    for i in range(len(kernels_name)):
        count_vett.append(0)
    
    for i in training_set:
        idx = i['kernel_id']
        sign = i['conf_sign']
        curr= kernelToFamily(kernels_name[idx]) + str(signSet[idx].index(sign))
        count_vett[idx] +=1
        routines_names.append({'kernel':curr, 'used': 0, 'sign' : sign})

    return routines_names

def genConfSet(kernels_name, training_set):
    signVet = []
    count_vett = []
    for i in range(len(kernels_name)):
        s = set()
        count_vett.append(s)

    for i in training_set: 
        idx = i['kernel_id']
        sign = i['conf_sign']
        count_vett[idx].add(sign)

    set_dim = 0 
    for i in range(len(kernels_name)):
        signVet.append(list(count_vett[i]))
        set_dim += len(count_vett[i])

    print ("Collapsed " , str( len(training_set) - set_dim) , " of ", str(len(training_set)), sep="")


    return signVet



def genConfSign(parameters):
    sign = ""
    for p in parameters:
        sign+= p + str(parameters[p])

    return sign


# Generate C++ source code of the Decision Tree
def genSourceCode(library_root_path, kernel_name, d_tree,training_set):
    #src/database/kernels

    if not os.path.exists(library_root_path):
    	print ("[ERROR] : invalid library_root_path")
    	exit(1)
    	


    inference_src_file = "dvdt_infer.hpp"
    tmp_file = open( inference_src_file, 'w')

   
    global header
    global close_fun
    global acc_fun
    global aux_fun_header
    global footer
    global output_dir
    global json_out_dir
    #Write class header stuff
    tmp_file.write(header)    

    #Inference function body
    tree_to_code(d_tree, getFeatureNames(), training_set['X'], tmp_file, training_set['W'])

    #Write close_fun
    tmp_file.write(close_fun)

    #Write class footer stuff
    tmp_file.write(footer) 
    
    tmp_file.close()
    #Copy the file in the library path
    copyfile(inference_src_file, library_root_path + os.sep + 'src' + os.sep + inference_src_file)
    
    # I want just one copy of each used kernel, even if it appears in more than one leafs
    used_list = []
    for r in training_set['W']:
        if r['used'] == 1:
            if r['sign'] in used_list:
                r['used'] = 0
            else:
                used_list.append(r['sign'])

    #Update the json file to the json directory
    updateKernelOnJson(training_set,output_dir,json_out_dir)
    #Generate .hpp files for the new routines
    db_script_path = library_root_path + os.sep + 'scripts' + os.sep + 'database'
    db_script = 'python ' + db_script_path + os.sep + 'database.py -v ' + json_out_dir + ' ' + library_root_path
    os.system( db_script)
    
    #Add the new routines to the database
    db_cpp = library_root_path + os.sep + 'src' + os.sep + 'database' + os.sep + 'database.cpp'
    db_cpp_file = open(db_cpp,'r+')
    db_content = db_cpp_file.read()
    db_content_row = db_content.split('\n')
    db_num_row = len(db_content_row)
    
    namespace_line = 'namespace clblast {'
    db_init_line = 'const std::vector<Database::DatabaseEntry> Database::database = std::vector<Database::DatabaseEntry>{'
    
    #Check the namespace_line and db_init_line are in the file
    if db_content_row.count(namespace_line) != 1 or db_content_row.count(db_init_line) != 1:
        print ("[FATAL] : invalid installation")
        exit(1)
    
    
    idx_namespace_line = db_content_row.index(namespace_line)
    for r in training_set['W']:
        if r['used'] == 1:
            inc_line = "#include \"database/kernels/" + r['kernel'] +".hpp\""
            db_content_row.insert(idx_namespace_line,inc_line)
    

    idx_db_init_line = db_content_row.index(db_init_line) + 1
    for r in training_set['W']:
        if r['used'] == 1:
            single = "database::"+ r['kernel'] +'Single, '
            line = single
            db_content_row.insert(idx_db_init_line,line)
    
    #Replace the old file content with the new one
    db_cpp_file.seek(0,0)
    for line in db_content_row:
        db_cpp_file.write("%s\n" %line)

    db_cpp_file.close()



def getIdx(array_value):
    for i in range(len(array_value)):
         if array_value[i] != 0:
             return i


from sklearn.tree import _tree
tree_height = 0

def tree_to_code(tree, feature_names, training_set, out_file, routines_name):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node, depth,selected_routines, out_file):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            
            name = feature_name[node]
            threshold = tree_.threshold[node]
            value=("if( " + str(name) + " <= " + str(threshold) + " ) {\n")
            # print "{}if {} <= {}:".format(indent, name, threshold)
            out_file.write(value)
            recurse(tree_.children_left[node], depth + 1, selected_routines,out_file)
            out_file.write("}\n")
            # print "{}else:  # if {} > {}".format(indent, name, threshold)
            
            value=("else {\n ")
            out_file.write(value)
            recurse(tree_.children_right[node], depth + 1, selected_routines,out_file)
            out_file.write("}\n")
        else:
            global tree_height
            if depth > tree_height:
                tree_height = depth
            # print tree_.value[node][0]
            # print getIdx(tree_.value[node][0])
            selected_routines.append(routines_name[getIdx(tree_.value[node][0])]['kernel'])
            # print "{}return {}".format(indent, training_set[getIdx(tree_.value[node][0])])
            value = ("\n routines_vett.push_back(\"" + routines_name[getIdx(tree_.value[node][0])]['kernel'] + "\");\n")
            # print value
            out_file.write(value)
            value = ("k_name = \"" + routines_name[getIdx(tree_.value[node][0])]['kernel'] + "\";")
            out_file.write(value)
            
            if 'direct' in routines_name[getIdx(tree_.value[node][0])]['kernel']:
                value = ("\n *flag=1;\n")
            else:
                value = ("\n * flag = 0;\n")

            routines_name[getIdx(tree_.value[node][0])]['used'] = 1
            out_file.write(value)
            out_file.write("#ifdef DVDT_DEBUG\n printf(\"" +  routines_name[getIdx(tree_.value[node][0])]['kernel'] +"\");\n#endif\n");

            global num_leaf
            num_leaf += 1

    selected_routines=[]
    recurse(0, 1,selected_routines,out_file)
    return selected_routines


def splitDataset(dataset, ratio):
    #ratio tra 0 e 100
    if ratio > 100 or ratio < 0 : 
        print("[ERROR] : invalid ratio")
        exit(1)
    random.seed(1)
    TRAINING = []
    TEST = []
    SET = []
    dataset_len = len(dataset['X'])
    if(dataset_len <= 100):
        training_dim = int(math.ceil(( (dataset_len-1) / 100.0 ) * ratio))
    else:
        training_dim = int(math.ceil(( dataset_len / 100.0 ) * ratio))
    test_dim = dataset_len - training_dim
    print (training_dim)
    print (test_dim) 
    print (dataset_len)
    idx_set=set()
    while len(idx_set) < training_dim:
        idx_set.add(random.randint(0,(dataset_len-1)))

    # print idx_set
    X=[]
    Y=[]
    W=[]
    Z=[]
    for i in idx_set:
        X.append(dataset['X'][i])
        Y.append(dataset['Y'][i])
        W.append(dataset['W'][i])
        Z.append(dataset['Z'][i])
    
    TRAINING = {'X' : X , 'Y' : Y, 'W' : W, 'Z' : Z}
    X=[]
    Y=[]
    W=[]
    Z=[]
    for i in range(dataset_len):
        if not i in idx_set:
            X.append(dataset['X'][i])
            Y.append(dataset['Y'][i])
            W.append(dataset['W'][i])
            Z.append(dataset['Z'][i])

    TEST = {'X' : X , 'Y' : Y, 'W' : W, 'Z' : Z}
    return {'TRAINING': TRAINING, 'TEST' : TEST}

def printTestDataset(test_set, out_file = '/tmp/test_set'):
    l = len(test_set['X'])
    f = open(out_file,'w')
    for i in range(l):
        f.write(str(test_set['X'][i]))
        f.write(',')
        f.write(str(test_set['Y'][i]))
        f.write('\n')

def printTestDatasetInfo(test_set, out_file = '/tmp/test_set.info'):
    l = len(test_set['X'])
    f = open(out_file,'w')
    for i in range(l):

        f.write(str(test_set['X'][i][0]))
        f.write(',')
        f.write(str(test_set['X'][i][1]))
        f.write(',')
        f.write(str(test_set['X'][i][2]))
        f.write('\n')



def copyDataset(src, dst):
    from distutils.dir_util import copy_tree
    copy_tree(src,dst)
################################################################################
################################################################################

# MAIN 
################################################################################
parser = argparse.ArgumentParser(description='Adaptive Library')

#parser.add_argument("--url", action = "store", dest = "training_url", help = "Crowdsourcing url ( ) for training data")
parser.add_argument("--clblast_root", action = "store", dest = "clblast_root", required = True, help = "absolute path to CLBlast root")
#parser.add_argument("--max_num_leafs", action = "store", type = int, dest = "max_leafs", help = "the DT depth will be set to log_2(max_leafs)")
parser.add_argument("--kernel", action = "store", dest = "kernel_name", nargs ='*', default = ["xgemm"], help = "kernel name(s) you want data train on")
parser.add_argument("--max_tree_depth", action = "store", type = int, dest = "tree_depth", help = "the maximum DT depth")
parser.add_argument("--quiet", action = "store_true", help = "It will suppress CK output")
parser.add_argument("--ratio", action = "store", dest = "ratio", help = "define the ratio between training and test sets (default 80:20 pareto)")
parser.add_argument("--tree_min_samples_leaf", action = "store", default = 1, help = "specify also the type with tree_min_samples_leaf_type")
parser.add_argument("--dataset_dir", action ="store", required = True, help = "the directory containing the dataset")
parser.add_argument("--output_dir", action ="store", required = True, help = "the output directory (if it exists and it is not empty, all the content will be deleted)")
parser.add_argument("--platform", action = "store", required = True)
myarg=parser.parse_args()

platform = myarg.platform
tree_min_samples_leaf = 0
if isinstance(myarg.tree_min_samples_leaf,float):
    tree_min_samples_leaf = float(myarg.tree_min_samples_leaf)
else:
    tree_min_samples_leaf = int(myarg.tree_min_samples_leaf)


if os.path.exists(myarg.output_dir) and os.path.isdir(myarg.output_dir):
    if os.listdir(myarg.output_dir):
        print("[INFO]: output_directory is not empty. The content will be removed")
        import shutil
        shutil.rmtree(myarg.output_dir)

output_dir = myarg.output_dir
json_out_dir = output_dir + "/json"


ratio = DEFAULT_RATIO 
if myarg.ratio != None:
    ratio = int(myarg.ratio)

pipeline_output = 'out' if myarg.quiet else 'con'
DATASET=createTrainingSet(myarg)

print("Training dataset len: " + str( len(DATASET['TRAINING']['X'])))
print("Test dataset len: " + str( len(DATASET['TEST']['X'])))
tree_depth = myarg.tree_depth if myarg.tree_depth > 0 else  None
d_tree=createDecisionTree(DATASET['TRAINING'],tree_depth, tree_min_samples_leaf)

mean_acc=1
if ratio != 100 :
    mean_acc = d_tree.score(DATASET['TEST']['X'], DATASET['TEST']['Y'])
    print ("Mean Accurancy - " , str(mean_acc))



clblast_root = myarg.clblast_root
if "src" not in clblast_root:
    clblast_root+= os.sep + 'src'
genSourceCode(clblast_root, myarg.kernel_name,	d_tree,DATASET['TRAINING'])

if not os.path.exists(output_dir + os.sep + "info"):
    os.makedirs(output_dir + os.sep + "info")
    
dumpTrainingToFile(DATASET, output_dir + os.sep + "info" + os.sep + 'test_'+str(ratio) + '_' + str(myarg.tree_depth) + '.json')
printTestDataset(DATASET['TEST'], output_dir + os.sep + "info" + os.sep + 'test_'+str(ratio) + '_' + str(myarg.tree_depth))
printTestDatasetInfo(DATASET['TEST'],output_dir + os.sep + "info" + os.sep +  'test_'+str(ratio) + '_' + str(myarg.tree_depth) + '.info')

f=open(output_dir + os.sep + "info" + os.sep +'statistics.info', 'w')

print ("*************** Statistics ***************")
f.write("*************** Statistics ***************")
f.write("\n")

print ("Dataset size : " , str( len(DATASET['TRAINING']['X']) + len(DATASET['TEST']['X'])),sep="")
f.write("Dataset size : " + str( len(DATASET['TRAINING']['X']) + len(DATASET['TEST']['X'])))
f.write("\n")

print ("Training dataset ratio : " , str(ratio))
f.write("Training dataset ratio : " + str(ratio))
f.write("\n")

print ("Decision tree  # leaves : " , str(num_leaf))
f.write("Decision tree  # leaves : " + str(num_leaf))
f.write("\n")

print ("Decision heigth : " , str(tree_height), sep="")
f.write("Decision heigth : " + str(tree_height))
f.write("\n")

signSet = genConfSet(myarg.kernel_name, DATASET['TRAINING']['Z'])

for i in range(len(myarg.kernel_name)):    
    print ("Unique configurations for [" , myarg.kernel_name[i] + "] : " , str(len(signSet[i])), sep="" )
    f.write("Unique configurations for [" + myarg.kernel_name[i] + "] : " + str(len(signSet[i])) )
    f.write("\n")

# print "*************** Building the library ***************"

#buildLibrary()


print ("*************** Calculate the Accurancy ***************")
print ("Mean Accurancy - " , str(mean_acc), sep="")

f.write("*************** Calculate the Accurancy ***************")
f.write("\n")
f.write("Mean Accurancy - " + str(mean_acc))
f.write("\n")
f.close()


