#!/usr/bin/python

import json
import argparse

def getAccuracy(default_results, real_results):
  
    results = []
    num_of_test = len(default_results)


    mean_diff = 0.0
    mean_square_diff = 0.0
    mean_ratio = 0.0
    for i in range(num_of_test):
        
        #Retrieve matrix size and PT 
        m_d= default_results[i]['m']
        n_d = default_results[i]['n']
        k_d = default_results[i]['k']
        gflops_d = default_results[i]['GFLOPS_R']
        gflops_d_t = default_results[i]['GFLOPS_T']

        m_r= real_results[i]['m']
        n_r = real_results[i]['n']
        k_r = real_results[i]['k']
        gflops_r = real_results[i]['GFLOPS_R']
        gflops_r_t = real_results[i]['GFLOPS_T']

        if m_d != m_r or n_d != n_r or k_d != k_r or gflops_d_t != gflops_r_t :
            print "[FATAL] : exit"
            print "Default " + str(m_d) + "," + str(n_d) + "," + str(k_d) + "," + str(gflops_d_t)
            print "Real " + str(m_r) + "," + str(n_r) + "," + str(k_r) + "," + str(gflops_r_t)
            
        # PT - PR
        diff = gflops_d - gflops_r
        mean_diff += diff

        #1 - (PR /PT)
        ratio = 1.0 - ( gflops_d / gflops_r)
        mean_ratio += ratio

        #MSE 
        square_diff = (gflops_d - gflops_r)**2
        mean_square_diff += square_diff
        results.append({
            'm' : m,
            'n' : n , 
            'k' : k, 
            'GFLOPS_R' : gflops_r, 
            'GFLOPS_D' : gflops_d, 
            'GFLOPS_T' : gflops_r_t,
            'diff' : diff, 
            'ratio' : ratio, 
            'sqare_diff': square_diff
        })

    mean_diff = mean_diff / num_of_test
    mean_ratio = mean_ratio / num_of_test
    mean_square_diff = mean_square_diff / num_of_test
    accuracy = {
        'num_of_test' : num_of_test, 
        'mean_diff' : mean_diff,
        'mean_ratio' : mean_ratio,
        'mean_square_diff' : mean_square_diff,
        'results' : results 
        }

    return accuracy

def getResults(input_file):
    f = open(input_file)
    j = json.load(f)
    return j['results']

parser = argparse.ArgumentParser(description = 'Real VS Default accuracy')
parser.add_argument('--default_json', action = 'store', required = True)
parser.add_argument('--real_json', action = 'store', required = True)
parser.add_argument('--out_json', action = 'store', required = True)

myarg=parser.parse_args()


def_results = getResults(myarg.default_json)

real_results = getResults(myarg.real_json)

out_results = getAccuracy(def_results, real_results)

out = open(myarg.out_json, 'w')
json.dump(out_results, out, indent = 4)
out.close()