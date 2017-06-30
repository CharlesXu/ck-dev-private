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





################################################################################
################################################################################

# GLOBAL VARIABLES
################################################################################

#

pipeline_output='con' 
program = 'clblast-tune-ml'
cmd_key = 'xgemm-fp32'
platform = 'odroid-xu3'
run=5
num_leaf = 0
output_dir =''
json_out_dir = ''

header = """
#ifndef DVDT_INFER_H
#define DVDT_INFER_H

#include <vector>

#include "clblast.h"

namespace clblast{

template <typename T> 
        const std::vector<std::string> GetConf(const Layout layout, const Transpose a_transpose,const Transpose b_transpose,
        const size_t m, const size_t n, const size_t k,
        const T alpha, const size_t a_offset, const size_t a_ld,
        const size_t b_offset, const size_t b_ld,
        const T beta, const size_t c_offset, const size_t c_ld, int *flag){

        std::vector<std::string> routines_vett = {"Copy","Pad","Transpose","Padtranspose","KernelSelection"};


        """



footer = """ 
return routines_vett;
}

template const std::vector<std::string> GetConf<float>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const float alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const float beta, const size_t c_offset, const size_t c_ld, int *flag);

template const std::vector<std::string> GetConf<double>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const double alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const double beta, const size_t c_offset, const size_t c_ld, int *flag);

template const std::vector<std::string> GetConf<float2>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const float2 alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const float2 beta, const size_t c_offset, const size_t c_ld, int *flag);

template const std::vector<std::string> GetConf<double2>(const Layout layout, const Transpose a_transpose,
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const double2 alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const double2 beta, const size_t c_offset, const size_t c_l, int *flag);

}
#endif
"""



################################################################################
################################################################################

# GENERATE TRAINING DATASET
################################################################################
# Generate Input Dataset
def generateInputDataset(num_samples=4):
    X=[]
    for i in range(num_samples):
        curr={'m': 2**(i+2), 'n': 2**(i+2) ,'k' : 2**(i+2)}
        X.append(curr)

    return X

def getLibraryRootPath(library_tags):
    ii={'action': 'search',
        'module_uoa': 'env',
        'tags' : library_tags
    }
    r = ck.access(ii)
    if r['return'] > 0:
    	print "[ERROR] : unable to find library_tags entry " + library_tags
    	return r
    return r

def runPipeline(data_uoa, cmd_key, env, cdeps, rdeps, training_set):
	 # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'out'}
    r=ck.access(ii)
    if r['return']>0: return r

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    # Load  program meta and desc to check deps.
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa': data_uoa}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']

     # Get compile-time and run-time deps.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})

    # # Merge rdeps with cdeps for setting up the pipeline (which uses
    # # common deps), but tag them as "for_run_time".
    for k in rdeps:
        cdeps[k]=rdeps[k]
        cdeps[k]['for_run_time']='yes'
    
    global pipeline_output
    ii={'action' : 'pipeline',
                
        'target_os':tos,
        'device_id':tdid,

        'module_uoa' : 'program',
        'data_uoa' : program,
        'cmd_key' : cmd_key,
        'prepare' : 'yes',
        'dependencies' : cdeps,
        'no_compiler_description' : 'yes',
        'out' : 'con',
        'no_state_check' : 'yes',
        'flags' : '-O3',
        'env':{
            'CK_CLBLAST_NUM_ITERATIONS': env['run'],
            'CK_TUNER_NUM_OF_STRATEGIES': env['num_of_strategy'],
            'CK_SEARCH_STRATEGY': env['search_strategy'],
            'CK_PSO_SWARM_SIZE':env['pso_swarm_size'],
            'CK_PSO_INF_G' : env['pso_inf_g'],
            'CK_PSO_INF_L' : env['pso_inf_l'],
            'CK_PSO_INF_R' : env['pso_inf_r']
           	       
           	},
        'cpu_freq':'max',
        'gpu_freq':'max'
        }
    r=ck.access(ii)
    
    if r['return']>0: return r
    fail=r.get('fail','')
    if fail=='yes': return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    ready=r.get('ready','')
    if ready!='yes': return {'return':11, 'error':'pipeline not ready'}


    state=r['state']
    tmp_dir=state['tmp_dir']
    xcdeps=r.get('dependencies',{})
    # Clean pipeline.
    if 'ready' in r: del(r['ready'])
    if 'fail' in r: del(r['fail'])
    if 'return' in r: del(r['return'])
    pipeline=copy.deepcopy(r)


    record_repo='local'
    record_uoa='create-training-dataset-' + cmd_key + '-' + platform
    ck.out('---------------------------------------------------------------------------------------')
    ck.out('Experiment - %s:%s' % (record_repo, record_uoa))
    
    size_m = []
    size_n = []
    size_k = []
    
    for j in range(len(training_set)):
        size_m.append(training_set[j]['m'])
        size_n.append(training_set[j]['n'])
        size_k.append(training_set[j]['k'])

    cpipeline=copy.deepcopy(pipeline)
    ii={
        'action':'autotune',
        'module_uoa':'pipeline',
        'data_uoa':'program',
        'env':{
            'CK_CLBLAST_NUM_ITERATIONS': env['run'],
            'CK_TUNER_NUM_OF_STRATEGIES': env['num_of_strategy'],
            'CK_SEARCH_STRATEGY': env['search_strategy'],
            'CK_PSO_SWARM_SIZE':env['pso_swarm_size'],
            'CK_PSO_INF_G' : env['pso_inf_g'],
            'CK_PSO_INF_L' : env['pso_inf_l'],
            'CK_PSO_INF_R' : env['pso_inf_r']
           	       
           	},
        'choices_order':[
            [
             '##env#CK_CLBLAST_MSIZE'
            ],
            [
             '##env#CK_CLBLAST_NSIZE',
            ],
            [
             '##env#CK_CLBLAST_KSIZE'
            ],
            [
             '##env#CK_CLBLAST_NUM_ITERATIONS'
            ],
            [
             '##env#CK_TUNER_NUM_OF_STRATEGIES'
            ],
            [
             '##env#CK_SEARCH_STRATEGY'
            ],
            [
             '##env#CK_PSO_SWARM_SIZE'
            ],
            [
             '##env#CK_PSO_INF_G'
            ],
            [
             '##env#CK_PSO_INF_G'
            ],
            [
             '##env#CK_PSO_INF_R'
            ]

        ],
        'choices_selection':[
            {"type":"loop-with-next", "choice":size_m, "default":"256"},
            {"type":"loop-with-next", "choice":size_n, "default":"256"},
            {"type":"loop-with-next", "choice":size_k, "default":"256"},
            {"type" : "loop", "choice":[env['run']] , 'default':[env['run']]},
            {"type" : "loop", "choice":[env['num_of_strategy']] , 'default':[env['num_of_strategy']]},
            {"type" : "loop", "choice":[env['search_strategy']] , 'default':[env['search_strategy']]},
            {"type" : "loop", "choice":[env['pso_swarm_size']] , 'default':[env['pso_swarm_size']]},
            {"type" : "loop", "choice":[env['pso_inf_g']] , 'default':[env['pso_inf_g']]},
            {"type" : "loop", "choice":[env['pso_inf_l']] , 'default':[env['pso_inf_l']]},
            {"type" : "loop", "choice":[env['pso_inf_r']] , 'default':[env['pso_inf_r']]}
        ],
        'features_keys_to_process':['##choices#*'],


        'iterations':-1,
        'repetitions':1,
        'record':'yes',
        'record_failed':'yes',
        'record_params':{
            'search_point_by_features':'yes'
        },
        'record_repo':record_repo,
        'record_uoa':record_uoa,
        'tags':['create-training-dataset', cmd_key, platform],
        'pipeline': cpipeline,
        'out':pipeline_output

    }
    r=ck.access(ii)
    
    if r['return']>0: 
        return r
    fail=r.get('fail','')
    if fail=='yes':
       return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}




    
def runProgram(data_uoa, cmd_key, env, cdeps, rdeps):
    ii={'action' : 'run',
                
        'target_os':tos,
        'device_id':tdid,

        'module_uoa' : 'program',
        'data_uoa' : program,
        'cmd_key' : cmd_key,

        'env':{
            'CK_CLBLAST_NUM_ITERATIONS': env['run'],
            'CK_CLBLAST_MSIZE': env['m'],
         	'CK_CLBLAST_NSIZE': env['n'],
            'CK_CLBLAST_KSIZE': env['k'],
            'CK_TUNER_NUM_OF_STRATEGIES': env['num_of_strategy'],
            'CK_SEARCH_STRATEGY': env['search_strategy'],
            'CK_PSO_SWARM_SIZE':env['pso_swarm_size'],
            'CK_PSO_INF_G' : env['pso_inf_g'],
            'CK_PSO_INF_L' : env['pso_inf_l'],
            'CK_PSO_INF_R' : env['pso_inf_r']
           	       
           	},
        'cpu_freq':'max',
        'gpu_freq':'max',
            

        }

    r_program=ck.access(ii)
    # ck.debug_out(r)
    if r_program['return']>0: 
        print "[ERROR] : run failed"

    return r_program['return']



# Tune the library and extract the best configurations 
# for each matrix in the Input Dataset
def tuneLibrary(training,output_dir,kernels_name):

	 # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'out'}
    r=ck.access(ii)
    if r['return']>0: return r

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    # Load  program meta and desc to check deps.
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa':program}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']

     # Get compile-time and run-time deps.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})

    # # Merge rdeps with cdeps for setting up the pipeline (which uses
    # # common deps), but tag them as "for_run_time".
    for k in rdeps:
        cdeps[k]=rdeps[k]
        cdeps[k]['for_run_time']='yes'
  
    training_dim = len (training)
    if training_dim <= 0 :
        print "[ERROR] : Invalid training set"
        return None
    Z = []
    ii={'action': 'search',
        'module_uoa': 'program',
        'data_uoa': program
    }
    r = ck.access(ii)
    if r['return'] > 0:
    	print "[ERROR] : unable to find program entry " + program
    	return r
    env ={ 
        	'run' : run,
        	'num_of_strategy' : 1,
        	'search_strategy' : 2,
        	'pso_swarm_size' : 8,
        	'pso_inf_g' : 0.3,
        	'pso_inf_l' : 0.6,
        	'pso_inf_r' : 0.1
        }
    for j in range(len(kernels_name)):
        cmd_key = kernels_name[j] + '-fp32'
        runPipeline(program, cmd_key, env, cdeps, rdeps, training)

    exp_dir=r['lst'][0]['path']
    exp_dir = exp_dir + '/tmp'
    for e in os.listdir(exp_dir):
        if kernels_name[0] in e and 'multiconf' in e :
            copyfile(exp_dir + os.sep + e , output_dir + os.sep + e)
    return 0




################################################################################
################################################################################

# GET TRAINING DATASET
################################################################################
def loadModelMatrixes(csv_files_dir):
    
    m = set()
    for i in os.listdir(csv_files_dir):
        f=open(csv_files_dir + os.sep + i)
        lines=f.read().split('\r\n')
        for j in range(2,len(lines)):
            m.add(lines[j])
        f.close()
    return m


def getFeatureNames():
    feature_names = ['m', 'n', 'k', 'm * n * k', 'kernel_name']
    return feature_names

def getTrainingFromUrl(training_url):
	print "NOT IMPLEMENTED YET"
	return generateInputDataset()

def getTrainingFromFile(training_file):
	json_data = open(training_file)
	TRAINING_SET = json.load(json_data)
	json_data.close()
	return TRAINING_SET

def getRandomMatrix(random_num=1, seed=None):
    X = []
    random.seed(seed)
    for i in range(random_num):
        m = 2 ** random.randint(1, 12)
        n = 2 ** random.randint(1, 12)
        k = 2 ** random.randint(1, 12)
        X.append({'m' : m, 'n' : n, 'k': k})

    return X

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
            print "[WARN] : The tuner was not able to find any valid configuration"
            print "[WARN] : Skipping the matrix"
            json_data.close()
            continue

        kernel = d['kernel_family']
        kernel = kernel[:-2]
        print "KERNEL : " + kernel
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

        print e
        print gflops
        kernel_id = getKernelId(kernels_array, kernel)
      	#print kernelId
        count +=1
        # NOTE : For now we use only m, n, k as features and gflops as labels
        X.append([ m,n,k,m*n*k ])
        #X.append({'m': m, 'n' : n , 'k' : k})
        Y.append(str(gflops))
        Z.append({
        	'm': m, 
        	'n' : n , 
        	'k' : k, 
        	'kernel': kernel,
        	'kernel_id' : kernel_id,
        	'parameters': parameters, 
        	'time': time, 
        	'precision' : precision,
        	'device': device,
        	'device_type' : device_type,
        	'device_vendor' : device_vendor,
        	'device_core_clock': device_core_clock,
        	'device_compute_units': device_compute_units,
        	'file_path' : e,
        	})
        json_data.close()

    T={'X' : X, 'Y' : Y, 'Z' : Z}
    return T


################################################################################
################################################################################

# DUMP TRAINING DATASET
################################################################################

def dumpTrainingToFile(training_set, output_file_path):
    out_file = open(output_file_path)
    json.dump(training_set,out_file)
    ou.close()

def treePlot(d_tree, output_file_path):
    out_file = open('/tmp/prova.dot','w')
    dot_data=tree.export_graphviz(d_tree, out_file=out_file)

    # dot_data= tree.export_graphviz(d_tree)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf(output_file_path)



def updateKernelOnJson(training_set, exp_dir, out_dir):
    for i in range(len(training_set['Z'])):
        e = training_set['Z'][i]
        f_in = open(exp_dir + os.sep + e['file_path'])
        json_data = json.load(f_in)

        
        json_data['kernel_family'] = training_set['W'][i]['kernel']

        f_out = open(out_dir + os.sep + e['file_path'],'w')
        json.dump(json_data,f_out)
        f_in.close()
        f_out.close()

def getRandomMatrixFromSet(dataset, num_samples, seed=None):
    X=[]
    random.seed(seed)
    dim=len(dataset)
    for i in range(num_samples):
        e=random.sample(dataset,1)
        e=e[0].split(',')
        X.append({'m' : e[0], 'n' : e[1], 'k': e[2]})

    return X

def getAllMatrixFromSet(dataset):
    X=[]
    for e in dataset:
        e=e.split(',')
        X.append({'m' : e[0], 'n' : e[1], 'k': e[2]})
    
    return X

# Create the Training Set
def createTrainingSet(arg):
    
    TRAINING_SET=[]
    
    if arg.training_url != None :
        TRAINING_SET = getTrainingFromUrl(arg.training_url)
        return TRAINING_SET

    if arg.fp != None:
    	TRAINING_SET = getTrainingFromFile(arg.fp)
    	return TRAINING_SET
    
    if arg.csv_files_dir != None:
        M = loadModelMatrixes(arg.csv_files_dir)
        #X = getRandomMatrixFromSet(M,5)
        X = getAllMatrixFromSet(M)
    elif arg.random_num != None:
        X = getRandomMatrix(arg.random_num,arg.seed)
    else:
        X = generateInputDataset()


    print X

    print "[INFO] : Training dataset len : " + str(len(X))
    global output_dir
    global json_out_dir
    output_dir = '/tmp/exp'
    if arg.output_dir != None:
        output_dir = arg.output_dir

    if not os.path.exists(output_dir):
    	os.makedirs(output_dir)
    else:
    	print "[INFO] : " + output_dir + " exists"
   
    json_out_dir = output_dir + '/json'
    if not os.path.exists(json_out_dir):
    	os.makedirs(json_out_dir)
    else:
    	print "[INFO] : " + json_out_dir + " exists"
   
    r = tuneLibrary(X,output_dir,arg.kernel_name)
    if r > 0:
        print "[FATAL] : exit"
        exit(1)


    TRAINING_SET = getTrainingFromDirectory(arg.kernel_name,output_dir)
    routines_names = generateRoutinesNames(arg.kernel_name,TRAINING_SET['Z'])

    TRAINING_SET['W'] = routines_names
    updateKernelOnJson(TRAINING_SET,output_dir,json_out_dir)
    print "X"
    print TRAINING_SET['X']
    print "Y"
    print TRAINING_SET['Y']
    print "W"
    print TRAINING_SET['W']
    print "Z"
    print TRAINING_SET['Z']
    
    return TRAINING_SET


################################################################################
################################################################################

# DECISION TREE
################################################################################



# BUILD A DECISION TREE
# (Using the training dataset)
def createDecisionTree(training_set,depth=None):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(training_set['X'],training_set['Y'])
    return clf

def chooseConf(d_tree, test_set):
    return d_tree.predict(test_set)

def kernelToFamily(kernel):
    k=re.sub(r'_\d+', '', kernel)
    k = k.split('_')
    f=''
    for i in k:
        f += i
    return f.capitalize()

def generateRoutinesNames(kernels_name, training_set):
    routines_names = []
    count_vett=[]
    
    for i in range(len(kernels_name)):
        count_vett.append(0)
    
    for i in training_set:
    	#print i
        idx = i['kernel_id']
        #print idx
        curr= kernelToFamily(kernels_name[idx]) + str(count_vett[idx])
        #curr= kernels_name[idx].capitalize() + str(count_vett[idx])
        count_vett[idx] +=1
        routines_names.append({'kernel':curr, 'used': 0})

    return routines_names



   

# Generate C++ source code of the Decision Tree
def genSourceCode(library_root_path, kernel_name, d_tree,training_set):
    #src/database/kernels

    if not os.path.exists(library_root_path):
    	print "[ERROR] : invalid library_root_path"
    	exit(1)

    
    # for e in os.listdir(library_root_path):
    #     if kernel_name == e[:-4]: # remove .hpp
    # 	    print e
    	


    inference_src_file = "dvdt_infer.hpp"
    tmp_file = open( inference_src_file, 'w')

   
    global header
    global footer
    global output_dir
    global json_out_dir
    #Write class header stuff
    tmp_file.write(header)    

    #Inference function body
    tree_to_code(d_tree, getFeatureNames(), training_set['X'], tmp_file, training_set['W'])

    #Write class footer stuff
    tmp_file.write(footer) 
    
    tmp_file.close()
    #Copy the file in the library path
    copyfile(inference_src_file, library_root_path + os.sep + 'src' + os.sep + inference_src_file)
    
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
        print "[FATAL] : invalid installation"
        exit(1)

    idx_namespace_line = db_content_row.index(namespace_line)
    for r in training_set['W']:
        if r['used'] == 1:
            inc_line = "#include \"database/kernels/" + r['kernel'] +".hpp\""
            db_content_row.insert(idx_namespace_line,inc_line)
    

    idx_db_init_line = db_content_row.index(db_init_line) + 1
    for r in training_set['W']:
        if r['used'] == 1:
            half = "database::"+ r['kernel'] +'Half, '
            single = "database::"+ r['kernel'] +'Single, '
            double = "database::"+ r['kernel'] +'Double, '
            cpx_single = "database::"+ r['kernel'] +'ComplexSingle, '
            cpx_double = "database::"+ r['kernel'] +'ComplexDouble,'
            line = half + single + double + cpx_single + cpx_double
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
            # print tree_.value[node][0]
            # print getIdx(tree_.value[node][0])
            selected_routines.append(routines_name[getIdx(tree_.value[node][0])]['kernel'])
            # print "{}return {}".format(indent, training_set[getIdx(tree_.value[node][0])])
            value = ("\n routines_vett.push_back(\"" + routines_name[getIdx(tree_.value[node][0])]['kernel'] + "\");\n")
            # print value
            out_file.write(value)
            if 'direct' in routines_name[getIdx(tree_.value[node][0])]['kernel']:
                value = ("\n *flag=1;\n")
            else:
                value = ("\n * flag = 0;\n")

            routines_name[getIdx(tree_.value[node][0])]['used'] = 1
            out_file.write(value)
            out_file.write("printf(\"" +  routines_name[getIdx(tree_.value[node][0])]['kernel'] +" \n\");")

            global num_leaf
            num_leaf += 1

    selected_routines=[]
    recurse(0, 1,selected_routines,out_file)
    return selected_routines


################################################################################
################################################################################

# MAIN 
################################################################################
default_clblast_root='/home/marco/CK_TOOLS/lib-clblast-tune-multiconf-master-gcc-6.2.0-linux-32/src'
parser = argparse.ArgumentParser(description='Adaptive Library')

parser.add_argument("--url", action = "store", dest = "training_url", help = "Crowdsourcing url ( ) for training data")
parser.add_argument("--file", action = "store", dest ="fp", help = "Json file containing dump of training data")
parser.add_argument("--random_samples", action = "store", type = int, dest = "random_num", help = "Number of random matrix sizes. The tuner will be launched on each matrix")
parser.add_argument("--output_dir", action = "store", dest = "output_dir", help = "output_dir to store tuner results over training data")
parser.add_argument("--clblast_root", action = "store", dest = "clblast_root", required = True, default = default_clblast_root, help = "absolute path to CLBlast root")
parser.add_argument("--max_num_leafs", action = "store", type = int, dest = "max_leafs", help = "the DT depth will be set to log_2(max_leafs)")
parser.add_argument("--target_os", action = "store", dest = "tos")
parser.add_argument("--device_id", action = "store", type = int, dest = "did")
parser.add_argument("--kernel", action = "store", dest = "kernel_name", nargs ='*', default = ["xgemm"], help = "kernel name(s) you want data train on")
parser.add_argument("--max_tree_depth", action = "store", type = int, dest = "tree_depth", help = "the maximum DT depth")
parser.add_argument("--seed", type = int, help = "You can specify the initial seed for reproducibility. It only works with --random_samples")
parser.add_argument("--quiet", action = "store_true", help = "It will suppress CK output")
parser.add_argument("--csv", action="store", dest ="csv_files_dir", help="load Model matrix sizes from csv")
myarg=parser.parse_args()


pipeline_output = 'out' if myarg.quiet else 'con'
TRAINING_SET=createTrainingSet(myarg)
tree_depth = ( int (myarg.tree_depth)) if (myarg.tree_depth != None) else None
d_tree=createDecisionTree(TRAINING_SET,tree_depth)

TEST_SET =[ [243,312,64,243*312*64], [ 2048, 128,  64,2048*128*64],[  128,  128, 2048,128*128*2048]]
idx = chooseConf(d_tree,TEST_SET)
# print "Prediction : [243,312,64],[ 2048, 128,  64],[  128,  128, 2048] " 
# print idx

treePlot(d_tree,'/tmp/prova.pdf')

kernel_name ="xgemm"

clblast_root = myarg.clblast_root if myarg.clblast_root != None else "/home/marco/CK_TOOLS/lib-clblast-tune-master-gcc-6.2.0-linux-32/src"
genSourceCode(clblast_root, myarg.kernel_name,	d_tree,TRAINING_SET)

