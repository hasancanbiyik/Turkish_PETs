from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, 
                          Trainer)
# from torch.utils.data import DataLoader

import numpy as np
import evaluate
import logging
import ast

import os
from typing import Union
import re

# for updating an external results .csv file
import pandas as pd
from sklearn.metrics import confusion_matrix

from transformers import set_seed
from transformers import EarlyStoppingCallback, IntervalStrategy

def run_trainer(trainfile: str, 
                testfile: str, 
                output_dir: str, 
                logger: Union[logging.Logger, None], 
                seed: int = 42) -> AutoModelForSequenceClassification:
    """Runs trainer

    Args:
        trainfile (str): Train file (.csv)
        testfile (str): Test file (.csv)
        output_dir (str): Output directory
        logger Union[logging.Logger, None], optional: Logger to use

    Returns:
        AutoModelForSequenceClassification: Trained model
    """
    
    # log = logging.getLogger(__name__) if logger is None else logger
    
    # sanity checks
    for i in [trainfile, testfile, output_dir]:
        assert os.path.exists(i), f"File/Directory {i} does not exist"
    
    # seed needs to be set before model instantiation for full reproducability of first run? (https://discuss.huggingface.co/t/multiple-training-will-give-exactly-the-same-result-except-for-the-first-time/8493)
    set_seed(42)
    
    # load model
    # log.info('loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(os.getenv('MODEL_NAME'))
    
    # log.info('loading dataset...')
    dataset = load_dataset("csv", data_files={"train": trainfile, 
                                              "test": testfile})
    
    # define tokenizer and tokenize datasets
    # log.info('loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('TOKENIZER_NAME'), max_length=512)
    tokenizer.model_max_length = 512  # for some reason, this is required for https://huggingface.co/castorini/afriberta_base

    # **************** ADDING SPECIAL TOKENS ****************** #
    special_tokens = ast.literal_eval(os.getenv('SPECIAL_TOKENS'))
    if (special_tokens != -1):
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer)) 
    # tokenizer.add_special_tokens(["[PET_BOUNDARY]"]) # a method to add one special token at a time?
    # ************************************************************* #
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # log.info('tokenizing datasets...')
    tokenized_datasets = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

    # **************** FREEZING LAYERS ****************** #
    layers_to_freeze = ast.literal_eval(os.getenv('LAYERS_TO_FREEZE'))
    if (layers_to_freeze != -1):
        for name, param in model.named_parameters():
            if (re.search(r'\d+', name)): # if this layer has a number (if it doesn't, re.search() returns None)
                if (int(re.search(r'\d+', name).group(0)) in layers_to_freeze):
                    param.requires_grad = False # freezes the layer
        # output layer statuses to make sure
        for name, param in model.named_parameters(): # prints out layers and frozen status
            print(name, param.requires_grad)
    # ********************************************************* #

    # define training args
    training_args = TrainingArguments(output_dir = output_dir, 
                                      evaluation_strategy = "epoch", 
                                      num_train_epochs = float(os.getenv('NUM_EPOCHS')),
                                      learning_rate = float(os.getenv('LEARNING_RATE')),
                                      per_device_train_batch_size = int(os.getenv('BATCH_SIZE')),
                                      per_device_eval_batch_size = int(os.getenv('BATCH_SIZE')),
                                      warmup_steps = int(os.getenv('WARMUP_STEPS')),
                                      logging_strategy = 'epoch',
                                      logging_first_step = True,
                                      metric_for_best_model = 'f1',
                                      save_strategy = 'epoch', # set to 'no' for no model saving during training
                                      save_total_limit = 1,
                                      load_best_model_at_end = True,
                                     )
    
    # define evaluation metrics
    metric_f1 = evaluate.load("f1")
    metric_pr = evaluate.load("precision")
    metric_re = evaluate.load("recall")
    # metric_acc = evaluate.load("accuracy")

    train_output = os.getenv('TRAIN_RESULTS_CSV')
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        f1 = metric_f1.compute(predictions=predictions, 
                               references=labels, 
                               average='macro')
        
        recall = metric_re.compute(predictions=predictions, 
                                   references=labels,
                                  average='macro')
        
        precision = metric_pr.compute(predictions=predictions, 
                                      references=labels,
                                     average='macro')
        
        # ***** update an external results .csv file ***** # 
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        if (os.path.basename(train_output) not in os.listdir(os.getenv('EXP_DIR'))):
            df = pd.DataFrame(columns=['test_no', 'f1', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp', 'preds'])
        else:
            df = pd.read_csv(train_output, index_col=0)
        file_no = re.search(r'\d+', os.path.basename(trainfile)).group(0)
        df.loc[len(df.index)] = [file_no, f1['f1'], precision['precision'], recall['recall'], tn, fp, fn, tp, predictions]
        df.to_csv(train_output)
        
        return f1
    
    # define test and train splits
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE')))]
    )

    trainer.train()

    # **************** POST-TRAINING EVALUATION PER LANGUAGE ****************** #
    # compute metrics 
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score

    # if evaluating, generate predictions using model and then return performance metrics
    if (bool(os.getenv('EVALUATION'))):
       
        test_output = os.getenv('TEST_RESULTS_CSV')
        
        def use_model_for_euph_predictions(tokenizer, model, test_data):
            preds = [] 
            # generate predictions    
            for i, row in test_data.iterrows():
                text = test_data.loc[i, 'text']
                inputs = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
                logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                preds.append(predicted_class_id)
            # compute metrics
            labels = test_data['label'].tolist()
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            precision = precision_score(labels, preds, average='macro')
            recall = recall_score(labels, preds, average='macro')
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            # print(preds)
            return accuracy, f1, precision, recall, tn, fp, fn, tp, preds
    
        print("Evaluating best model on test sets...")
        # prepare to evaluate on each of the languages' test sets
        TEST_NUM = re.search(r'\d+', os.path.basename(trainfile)).group(0)
        LANGS = ast.literal_eval(os.getenv('LANGS')) # ['chinese', 'english', 'spanish', 'yoruba']
        results = []
        
        # output result to an external .csv file
        if (os.path.basename(test_output) not in os.listdir(os.getenv('EXP_DIR'))):
            df = pd.DataFrame(columns=['TEST_NUM', 'LANG', 'accuracy', 'f1', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp', 'preds'])
        else:
            df = pd.read_csv(test_output, index_col=0)
        
        # BULK TESTING MODIFIER
        bulk_modifier = int(os.getenv('BULK_TESTING_MODIFIER'))
        if (bulk_modifier != -1):
            TEST_NUM = str(int(TEST_NUM) % bulk_modifier)
        
        # evaluate for each of the languages' test sets
        for lang in LANGS:
            test_data = pd.read_csv("{}/test_{}_{}.csv".format(os.getenv('EXP_DIR'), TEST_NUM, lang), index_col=0) # language-specific test file; a CSV of test examples
            result = use_model_for_euph_predictions(tokenizer, model, test_data)
            # update the external .csv file
            df.loc[len(df.index)] = [TEST_NUM, lang, result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]]
            df.to_csv(test_output)
    
    # save model
    if (os.getenv('SAVE_MODELS') == 'True'):
        trainer.save_model(output_dir)
    
    return model
