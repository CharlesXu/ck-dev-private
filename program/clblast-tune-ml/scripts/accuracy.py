#!/usr/bin/python

# Author : Marco Cianfriglia



################################################################################
################################################################################

# IMPORTS
################################################################################

import ck.kernel as ck
import os
import argparse





#Per il client il file è tmp-ck-clblast-client.json
# e la chiave per i GFLOPS : processed_gflops
# execution_time, m , n , k 

platform = ''



def runPipeline(data_uoa, cmd_key, m,n,k, library_uid):
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
    
    # Load the library speicified by the library_uid parameter
    ii = {
        'action' : 'load',
        'module_uoa' : 'env',
        'data_uoa' : library_uid
    }
    r = ck.access(ii)
    if r['return'] > 0 :
        print "[ERROR] : invalid library uid provided"
        return r

    cdeps['lib-clblast'] = library_uid
    ii={'action' : 'pipeline',
                
        'target_os':tos,
        'device_id':tdid,

        'module_uoa' : 'program',
        'data_uoa' : data_uoa,
        'cmd_key' : cmd_key,
        'prepare' : 'yes',
        'dependencies' : cdeps,
        'no_compiler_description' : 'yes',
        'out' : 'con',
        'no_state_check' : 'yes',
        'flags' : '-O3',
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

    ck.out('---------------------------------------------------------------------------------------')
   
    run =['5']
    size_m = [m]
    size_n = [n]
    size_k = [k]
    
    # size_m.append(m)
    # size_n.append(n)
    # size_k.append(k)

    cpipeline=copy.deepcopy(pipeline)
    ii={
        'action':'autotune',
        'module_uoa':'pipeline',
        'data_uoa':'program',
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
            ]
        ],
        'choices_selection':[
            {"type":"loop-with-next", "choice":size_m, "default":"256"},
            {"type":"loop-with-next", "choice":size_n, "default":"256"},
            {"type":"loop-with-next", "choice":size_k, "default":"256"},
            {"type" : "loop", "choice":run , 'default': 5}
        ],
        'features_keys_to_process':['##choices#*'],


        'iterations':-1,
        'repetitions':1,
        'record':'no',
        'pipeline': cpipeline,
        'out':'con'

    }
    r=ck.access(ii)
    
    if r['return']>0: 
        return r
    fail=r.get('fail','')
    if fail=='yes':
       return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}



def dvdt_accuracy(test_set,library_uid):

       
    # Se non è installata errore
    data_uoa = 'clblast-tune'
    cmd_key = 'default'
    ii={'action': 'search',
        'module_uoa': 'program',
        'data_uoa': data_uoa
    }
    r = ck.access(ii)
    if r['return'] > 0:
        print "[ERROR] : unable to find program entry " + data_uoa
        return r
    
    # Retrieve the experiment directory    
    exp_dir=r['lst'][0]['path']
    exp_dir = exp_dir + os.sep + 'tmp'
    exp_file = exp_dir + os.sep + 'tmp-ck-clblast-client.json'
    #Utilizzo default di program:clblast-tune selezionando la libreria library_tags

    
    results = []
    test_len = len(test_set)
    # per ogni tripla (m,n,k) del test_set
    # giro il program e prendo i gflops ottenuti
    # calcolo le varie metriche :
    # - differenza tra performance teoriche (PT) (tuner+eventuali kernel) e performance reale (PR) ottenuta con la libreria)
    # - differenza tra 1 - (rapporto tra  PR e PT ) 
    # - Mean square error  =  1/n * sum_{i=1}^{N} (PR - PT )^2
    # - media pesata della differenze dei punti di sopra

    mean_diff = 0.0
    mean_square_diff = 0.0
    mean_ratio = 0.0
    for i in range(test_len):
        
        #Retrieve matrix size and PT 
        m = test_set[i]['m']
        n = test_set[i]['n']
        k = test_set[i]['k']
        gflops_t = test_set[i]['gflops']

        #eseguo test PR
        runPipeline(data_uoa, cmd_key, m,n,k, library_uid):
        #estraggo GFLOPS_REALI 
        j_pr=json.load(open(exp_file))
        gflops_r = j_pr['processed_gflops']
        m_r = j_pr['m']
        n_r = j_pr['n']
        k_r = j_pr['k']
        if m_r != m or n_r != n or k_r != k:
            print "[FATAL] : the output file is not valid - invalid matrix size"
            exit(1)
        
        # PT - PR
        diff = gflops_t - gflops_r
        mean_diff += diff

        #1 - (PR /PT)
        ratio = 1.0 - ( gflops_r / gflops_t)
        mean_ratio += ratio

        #MSE 
        square_diff = (gflops_t - gflops_r)**2
        mean_square_diff += square_diff
        results.append({
            'm' : m,
            'n' : n , 
            'k' : k, 
            'GFLOPS_R' : gflops_r, 
            'GFLOPS_T' : 'gflops_t', 
            'diff' : diff, 
            'ratio' : ratio, 
            'sqare_diff': square_diff
        })

    accuracy = {
        'num_of_test' : test_len, 
        'mean_diff' : mean_diff,
        'mean_ratio' : mean_ratio,
        'mean_square_diff' : mean_square_diff,
        'results' : results 
        }

    return accuracy



parser = argparse.ArgumentParser(description='Dvdt Accuracy')


parser.add_argument('--library_uid', action = "store", required = True, help = "clblast multiconf library uid")
parser.add_argument('--dataset_json', action ="store", required= True)
parser.add_argument("--platform", action = "store", required = True)

myarg=parser.parse_args()

platform = myarg.platform
dataset = json.load(open(myarg.dataset_json))

acc = dvdt_accuracy(dataset['TEST']['Z'], myarg.library_uid)

print acc