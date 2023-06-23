#!/usr/bin/env python
# coding: utf-8

import os
import math
import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import clear_output

import warnings
import lifelines
from lifelines.utils import CensoringType
from lifelines.utils import concordance_index

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from formulaic.errors import FactorEvaluationError
import zstandard
import pickle
import yaml
import ray
import argparse

from datetime import date

def parse_args():
    parser=argparse.ArgumentParser(description="A script to calculate cindex for endpoints")
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--partition', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    args=parser.parse_args()
    return args

def save_pickle(data, data_path):
    with open(data_path, "wb") as fh:
        cctx = zstandard.ZstdCompressor()
        with cctx.stream_writer(fh) as compressor:
            compressor.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            
def load_pickle(fp):
    with open(fp, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as decompressor:
            data = pickle.loads(decompressor.read())
    return data

def load_data():
    base_path = "/sc-projects/sc-proj-ukb-cvd"
    print(base_path)

    project_label = "22_retina_phewas"
    project_path = f"{base_path}/results/projects/{project_label}"
    figure_path = f"{project_path}/figures"
    output_path = f"{project_path}/data"
    
    today = '230426'
    #today = str(date.today())
        
    experiment = '230426'
    experiment_path = f"{output_path}/{experiment}"
    pathlib.Path(experiment_path).mkdir(parents=True, exist_ok=True)
    
    in_path = f"{experiment_path}/coxph/predictions"
    out_path = f"{experiment_path}/benchmarks"
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    prediction_paths = pd.read_feather(f"{experiment_path}/prediction_paths.feather")
    #prediction_paths = os.listdir(in_path) 

    #endpoints_md = pd.read_csv(f"{experiment_path}/endpoints.csv")
    #endpoints = sorted(endpoints_md.endpoint.to_list())
    all_endpoints = sorted([l.replace('_prevalent', '') for l in list(pd.read_csv(f'/sc-projects/sc-proj-ukb-cvd/results/projects/{project_label}/data/{today}/endpoints.csv').endpoint.values)])
    endpoints_not_overlapping_with_preds = []
    endpoints = []
    for c in all_endpoints:
        if c not in endpoints_not_overlapping_with_preds: 
            endpoints.append(c)
    
    scores = [
        'Age+Sex', 'Retina', 'Age+Sex+Retina',
        "SCORE2", "SCORE2+Retina", 
        "ASCVD", "ASCVD+Retina", 
        "QRISK3", "QRISK3+Retina"
    ]
    
    eligable_eids = pd.read_feather(f"{output_path}/eligable_eids_{today}.feather")
    eids_dict = eligable_eids.set_index("endpoint")["eid_list"].to_dict()

    return output_path, experiment_path, in_path, out_path, endpoints, scores, prediction_paths, eids_dict

def read_single_partition_single_model(in_path, prediction_paths, endpoint, score, partition, model, time):
    paths = prediction_paths.query("endpoint==@endpoint").query("score==@score").query("partition==@partition").query("model==@model").path.to_list()
    data_preds = pd.DataFrame({})
    for path in paths:
        data_preds = pd.concat([data_preds, pd.read_feather(f"{in_path}/{path}", columns=["eid", f"Ft_{time}"])])
    data_preds = data_preds.set_index("eid").sort_index()
    data_preds.columns = ["Ft"]
    return data_preds

def read_partitions_single_model(in_path, prediction_paths, endpoint, score, model, time):
    paths = prediction_paths.query("endpoint==@endpoint").query("score==@score").query("model==@model").path.to_list()
    data_preds = pd.DataFrame({})
    #print('Calculating for: ', endpoint, score, model)
    for path in paths:
        data_preds = pd.concat([data_preds, pd.read_feather(f"{in_path}/{path}", columns=["eid", f"Ft_{time}"])])
    #print(data_preds.head())
    data_preds = data_preds.set_index("eid").sort_index()
    data_preds.columns = ["Ft"]
    return data_preds

def read_partitions(in_path, prediction_paths, endpoint, score, time):
    paths = prediction_paths.query("endpoint==@endpoint").query("score==@score").path.to_list()
    #paths = [p for p in prediction_paths if endpoint in p and score in p]
    data_preds = pd.DataFrame({})
    for path in paths:
        data_preds = pd.concat([data_preds, pd.read_feather(f"{in_path}/{path}", columns=["eid", f"Ft_{time}"])])
    data_preds = data_preds.set_index("eid").sort_index()
    data_preds.columns = ["Ft"]
    return data_preds

def prepare_data(in_path, prediction_paths, endpoint, score, partition, model, t_eval, output_path):
    # benchmark all models and all partitions
    #temp_preds = read_partitions(in_path, prediction_paths, endpoint, score, t_eval)
    # benchmark per model and per partition
    #temp_preds = read_single_partition_single_model(in_path, prediction_paths, endpoint, score, partition, model, t_eval)
    # benchmark all partitions per model
    temp_preds = read_partitions_single_model(in_path, prediction_paths, endpoint, score, model, t_eval)
    
    temp_tte = pd.read_feather(f"{output_path}/baseline_outcomes_220627.feather", 
        columns= ["eid", f"{endpoint}_event", f"{endpoint}_time"]).set_index("eid")
    temp_tte.columns = ["event", "time"]
    temp_data = temp_preds.merge(temp_tte, left_index=True, right_index=True, how="left")
    
    condition = (temp_data['event'] == 0) | (temp_data['time'] > t_eval)
    
    temp_data["event"] = (np.where(condition, 0, 1))
    
    temp_data["time"] = (np.where(condition, t_eval, temp_data['time']))

    return temp_data

from lifelines.utils import concordance_index

def calculate_cindex(in_path, prediction_paths, endpoint, score, partition, model, time, iteration, eids_i, output_path):  
    temp_data = prepare_data(in_path, prediction_paths, endpoint, score, partition, model, time, output_path)
    temp_data = temp_data[temp_data.index.isin(eids_i)]
    
    del eids_i
    
    try:
        cindex = 1-concordance_index(temp_data["time"], temp_data["Ft"], temp_data["event"])
    except ZeroDivisionError: 
        cindex=np.nan
    
    del temp_data
    
    # benchmark all models and all partitions
    #return {"endpoint":endpoint, "score": score, "iteration": iteration, "time":time, "cindex":cindex}
    # benchmark per model and per partition
    #return {"endpoint":endpoint, "score": score, "partition": partition, "model": model, "iteration": iteration, "time":time, "cindex":cindex}
    # benchmark all partitions per model
    return {"endpoint":endpoint, "score": score, "model": model, "iteration": iteration, "time":time, "cindex":cindex}

@ray.remote
def calculate_iteration(in_path, prediction_paths, endpoint, scores, partition, model, time, iteration, eids_e, output_path):  

    dicts = []
    valid = False
    grace = 100
    i = 0
    while valid==False:
        i+=1
        eids_i = np.random.choice(eids_e, size=len(eids_e)) if iteration != 0 else eids_e

        for score in scores:
            score_dict = calculate_cindex(in_path, prediction_paths, endpoint, score, partition, model, time, iteration, eids_i, output_path)
            if not np.isnan(score_dict['cindex']) or i>=grace:
                dicts.append(score_dict)
            else:
                valid = False
                dicts = []
                break

        if len(dicts) == len(scores) or i>=grace:
            valid=True
    return dicts
 
def main(args):

    # prepare env variables and initiate ray
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

    ray.init(address="auto")

    # read iteration and set seed
    iteration=args.iteration
    partition=args.partition
    model=args.model
    np.random.seed(iteration)

    # prepare setup
#     today = str(date.today())
    today = '230426'
    t_eval = 10
    
    # benchmark all models and all partitions
    #name = f"benchmark_cindex_{today}_iteration_{iteration}"
    # benchmark per model and per partition
    #name = f"benchmark_cindex_{today}_partition_{partition}_model_{model}_iteration_{iteration}"
    # benchmark all partitions per model
    name = f"benchmark_cindex_{today}_model_{model}_iteration_{iteration}"

    # load data
    output_path, experiment_path, in_path, out_path, endpoints, scores, prediction_paths, eids_dict = load_data()

    rows_ray = []
    for endpoint in endpoints: 
        eids_e = eids_dict[endpoint]
           
        ds = calculate_iteration.remote(in_path, prediction_paths, endpoint, scores, partition, model, t_eval, iteration, eids_e, output_path) #ray
        rows_ray.append(ds)

        del eids_e

    rows = [ray.get(r) for r in tqdm(rows_ray)] # ray
    #rows = rows_ray # not ray
    rows_finished = [item for sublist in rows for item in sublist]
    benchmark_endpoints = pd.DataFrame({}).append(rows_finished, ignore_index=True)
    
    #print('benchmark endpoints feather len:', len(benchmark_endpoints), flush=True)
    
    pathlib.Path(f"{experiment_path}/benchmarks/{today}").mkdir(parents=True, exist_ok=True)
    
    benchmark_endpoints.to_feather(f"{experiment_path}/benchmarks/{today}/{name}.feather")
    
    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    main(args)