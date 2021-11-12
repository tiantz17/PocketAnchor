import os
import time
import json
import pickle
import socket
import logging
import argparse
from importlib import import_module

import numpy as np
import torch

from torch.utils.data import DataLoader

class PocektAnchorPrediction(object):
    """
    PocketAnchor prediction using trained model
    """
    def __init__(self, args):
        """ common parameters """
        self.seed = args.seed
        self.info = args.info
        self.gpu = args.gpu
        self.use_cuda = args.gpu != "-1"
        self.path = args.path
        self.num_workers = args.num_workers

        """ special parameters """
        self.task = args.task
        self.dataset = args.dataset
        self.model = args.model
        self.model_path = args.model_path

        """ modules """
        self.DATASET = import_module(self.task+".src."+self.dataset)
        self.MODEL = import_module(self.task+".src."+self.model)

        """ training parameters """      
        self.dataset_params = self.DATASET.DATASET_PARAMS
        self.model_params = self.MODEL.MODEL_PARAMS
        self.train_params = self.MODEL.TRAIN_PARAMS

        if len(args.dataset_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.dataset_params.split(',')}
        else:
            update_params = {}
        self.dataset_params.update(update_params)

        if len(args.model_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.model_params.split(',')}
        else:
            update_params = {}
        self.model_params.update(update_params)

        if len(args.train_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.train_params.split(',')}
        else:
            update_params = {}
        self.train_params.update(update_params)

        """ update common parameters"""
        self.list_task = self.train_params["list_task"]

        """ local directory """
        file_folder = "PocketAnchorPrediction_task_{}_dataset_{}_model_{}_info_{}_{}_cuda{}"
        file_folder = file_folder.format(self.task, self.dataset, self.model, \
            self.info, socket.gethostname(), self.gpu)
        file_folder += time.strftime("_%Y%m%d_%H%M%S/", time.localtime())
        self.save_path = self.path + self.task + "/results/" + file_folder
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.define_logging()
        logging.info("Local folder created: {}".format(self.save_path))

        """ save hyperparameters """
        self.save_hyperparameter(args)

    def define_logging(self):
        # Create a logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.save_path + "logging.log",
            filemode='w')
        # Define a Handler and set a format which output to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def save_hyperparameter(self, args):
        args.dataset_params = self.dataset_params
        args.model_params = self.model_params
        args.train_params = self.train_params
        json.dump(dict(args._get_kwargs()), open(self.save_path + "config", "w+"), indent=4, sort_keys=True)

    def load_data(self):
        logging.info("Loading data...")
        # load data       
        self.Dataset = self.DATASET.DataSet(self.path + self.task + "/data/", self.dataset_params)

        self.Dataloader = DataLoader(self.Dataset, 
                                     batch_size=int(self.train_params["batch_size"]), 
                                     shuffle=False, 
                                     collate_fn=eval("self.DATASET.batch_data_process_"+self.model), 
                                     num_workers=self.num_workers, 
                                     drop_last=False, 
                                     pin_memory=self.use_cuda)
        
    def get_data_batch(self, batch_items):
        if self.use_cuda: 
            batch_items = [item.to(next(self.Model.parameters()).device) if item is not None and not isinstance(item, list) else \
                [it.to(next(self.Model.parameters()).device) for it in item] if isinstance(item, list) else \
                None for item in batch_items]

        return batch_items  

    def get_label_batch(self, batch_items):
        if self.use_cuda: 
            for key in batch_items.keys():
                if key in self.list_task:
                    batch_items[key] = batch_items[key].to(next(self.Model.parameters()).device)

        return batch_items

    def load_model(self, model_file):
        logging.info("Loading model...")
        # load model
        if self.use_cuda:
            device = torch.device("cuda:"+self.gpu)
        else:
            device = torch.device("cpu")
        self.Model = self.MODEL.Model(self.model_params)
        self.Model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
        # self.Model = self.Model.to(device)
        
        # load optimizer
        self.Model.load_optimizer(self.train_params)

    def predict(self):
        logging.info("Start prediction")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        list_results = []
        list_dict_collect = []
        self.load_data()
        list_models = []
        for i in os.listdir(self.model_path):
            if "best_model" not in i:
                continue
            if i[-2:] != "pt":
                continue
            list_models.append(i)
        list_models = sorted(list_models)

        for repeat, model_file in enumerate(list_models):
            self.load_model(self.model_path + model_file)
            dict_collect, results = self.evaluate()
            list_results.append(results)
            list_dict_collect.append(dict_collect)
            logging.info("="*60)
            logging.info("Repeat: {}, model: {}".format(repeat, model_file))
            for term in results: 
                logging.info("{}: {}".format(term, results[term]))
        results_all = self.merge_list_results(list_results)
        self.save(list_dict_collect, results_all)

        # ensemble
        results_ensemble = {}
        try:
            for task in list_dict_collect[0].keys():
                pred = np.concatenate([np.array(res[task]['pred']) for res in list_dict_collect], axis=1)
                label = np.concatenate([np.array(res[task]['label']) for res in list_dict_collect], axis=1)
                pred_ensemble = np.mean(pred, 1)
                label_ensemble = label[:,0]
                score_func = self.Dataset.score(self.Model.task_eval[task])
                results_ensemble[task] = score_func(pred_ensemble, label_ensemble)
        except:
            pass

        logging.info("="*60)
        logging.info("All done prediction at {}".format(self.dataset))
        logging.info("Results per repeat:")
        for repeat , results in enumerate(list_results):
            logging.info("="*60)
            logging.info("Repeat: {}".format(repeat))
            for term in results: 
                logging.info("{}: {}".format(term, results[term]))
        logging.info("-"*60)
        logging.info("Results ensemble:")
        for term in results_ensemble: 
            logging.info("{}: {}".format(term, results_ensemble[term]))
        logging.info("-"*60)
        logging.info("Results all:")
        for term in results_all: 
            logging.info("{}: {}".format(term, results_all[term]))

    def merge_list_results(self, list_results):
        dict_one_repeat = {}
        for results in list_results:
            for term in results:
                if term not in dict_one_repeat:
                    dict_one_repeat[term] = {}
                for score in results[term]:
                    if score not in dict_one_repeat[term]:
                        dict_one_repeat[term][score] = []
                    if isinstance(results[term][score], list):
                        dict_one_repeat[term][score].append(results[term][score][0])
                    else:
                        dict_one_repeat[term][score].append(results[term][score])
        for term in dict_one_repeat:
            for score in dict_one_repeat[term]:
                average = float(np.nanmean(dict_one_repeat[term][score]))
                std = float(np.nanstd(dict_one_repeat[term][score]))
                dict_one_repeat[term][score] = [average, std]

        return dict_one_repeat

    def evaluate(self):
        self.Model.eval()
        dict_collect = self.get_results_template()
        with torch.no_grad():
            for _, data in enumerate(self.Dataloader):
                data_tuple, label_dict = data
                data_tuple = self.get_data_batch(data_tuple)
                label_dict = self.get_label_batch(label_dict)
                pred_dict = self.Model(*data_tuple)

                for task in dict_collect:
                    dict_collect[task]["pred"].extend(pred_dict[task].cpu().data.numpy())
                    if task in label_dict:
                        dict_collect[task]["label"].extend(label_dict[task].cpu().data.numpy())

                if self.info == "debug":
                    break  

            results = {}
            for term in dict_collect:
                if term in label_dict:
                    score_func = self.Dataset.score(self.Model.task_eval[term])
                    results[term] = score_func(dict_collect[term]["pred"], dict_collect[term]["label"])

        return dict_collect, results

    def get_results_template(self):
        results = {}
        for task in self.list_task:
            results[task] = {"pred":[], "label":[]}
        return results

    def save(self, dict_collect, results):
        pickle.dump(dict_collect, open(self.save_path + "dict_collect", "wb"))
        json.dump(results, open(self.save_path + "results", "w"), indent=4, sort_keys=True)
        logging.info("Prediction results saved at " + self.save_path)


def main():
    parser = argparse.ArgumentParser()
    # define environment
    parser.add_argument("--gpu", default="0", help="which GPU to use", type=str)
    parser.add_argument("--seed", default=1234, help="random seed", type=int)
    parser.add_argument("--info", default="test", help="output folder special marker", type=str)
    parser.add_argument("--path", default="./", help="data path", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)

    # define task
    parser.add_argument("--task", default="Affinity", help="task", type=str)
    parser.add_argument("--dataset", default='CASF', help="dataset", type=str)
    parser.add_argument("--model", default="PocketAnchor", help="model", type=str)
    parser.add_argument("--setting", default="original", help="setting", type=str)
    parser.add_argument("--model_path", default="", help="path to saved model", type=str)

    # define parameters
    parser.add_argument("--dataset_params", default="", help="dict of dataset parameters", type=str)
    parser.add_argument("--model_params", default="", help="dict of model parameters", type=str)
    parser.add_argument("--train_params", default="", help="dict of training parameters", type=str)
    args = parser.parse_args()

    """ check """
    # task
    if not os.path.exists(args.path + args.task):
        raise NotImplementedError("Task {} not found!".format(args.task))
    
    # dataset
    if not os.path.exists(args.path + args.task + "/src/" + args.dataset + ".py"):
        raise NotImplementedError("Dataset {} not found!".format(args.dataset))
    
    # model
    if not os.path.exists(args.path + args.task + "/src/" + args.model + ".py"):
        raise NotImplementedError("Model {} not found!".format(args.model))
    
    # setting
    if args.setting == 'original':
        args.model_path = args.path + args.task + '/models/originalCASF/'
        args.dataset_params = 'datatype:test'
    elif args.setting == 'newprotein':
        args.model_path = args.path + args.task + '/models/newproteinCASF/'
        args.dataset_params = 'datatype:test'
    elif args.setting == 'expanded':
        args.model_path = args.path + args.task + '/models/newproteinCASF/'
        args.dataset_params = 'datatype:expand'
    else:
        raise NotImplementedError

    if args.task == 'PocketDetection':
        args.model_path = args.path + args.task + '/models/PDBbind2020/'
        
    """ claim class instance """
    PocektAnchor = PocektAnchorPrediction(args)
        
    """ Train """
    PocektAnchor.predict()


if __name__ == "__main__":
    main()

