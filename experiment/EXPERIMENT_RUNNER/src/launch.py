import os
import logging
import time
from datetime import datetime
from train import run_trainer

####################################################################
''' (0) SET EXPERIMENTAL DIRECTORY '''
####################################################################
EXP_DIR = "../Multilingual_Experiments/sample_experiment" # filepath to directory containing train/val/test splits; outputs (e.g. metrics and saved models) will go here

####################################################################
''' (1) CONFIGURE EXPERIMENT SETTINGS (AS ENVIRONMENT VARIABLES) '''
####################################################################

# NOTES: Consider using ast.literal_eval() in train.py to interpret environment variables and re-doing default values to empty lists/booleans -> False, for example
# function to set default values for optional settings? 

# model names
os.environ['MODEL_NAME'] = "xlm-roberta-base" # 'dccuchile/bert-base-spanish-wwm-cased' #  'xlm-roberta-large' , 'bert-base-multilingual-cased'
os.environ['TOKENIZER_NAME'] = "xlm-roberta-base" # 'dccuchile/bert-base-spanish-wwm-cased' #  'xlm-roberta-large' , 'bert-base-multilingual-cased'
os.environ['EXP_DIR'] = EXP_DIR # leave this as is
os.environ['SAVE_MODELS'] = 'True' # if True, will output best models for each trial into a folder called "saved_models" in EXP_DIR

# personal note created in EXP_DIR
os.environ['LOG'] = 'in this experiment i am doing something interesting'
# os.environ['EMAIL_NOTIFY'] = # '-1' # still developing this feature

# numbers in the first and last filenames (in `corpora` folder) of the experiment; set in `manifest.py`
os.environ['START_FILENUM'] = "0"
os.environ['END_FILENUM'] = "20" 

# some of the arguments in TrainingArguments that are commonly adjusted for fine-tuning in `train.py`
os.environ['NUM_EPOCHS'] = '30'
os.environ['LEARNING_RATE'] = '1e-5' 
os.environ['WARMUP_STEPS'] = '0' # '500' # default: 0
os.environ['BATCH_SIZE'] = '4'

# training settings
os.environ['TRAIN_RESULTS_CSV'] = f'{EXP_DIR}/results_train.csv' # filename of .csv where training and validation metrics will be output by `train.py`
os.environ['EARLY_STOPPING_PATIENCE'] = "5" # a callback variable for custom Trainer behavior controlling Patience
os.environ['LAYERS_TO_FREEZE'] = '-1' # '[0,1,2,3,4,5,6,7]' # '-1'
os.environ['SPECIAL_TOKENS'] = '-1' # '[\'[PET]\']' # '-1' # '[\'[PET_START]\', \'[PET_END]\']'

# evaluation settings
os.environ['EVALUATION'] = 'True' # whether to run evaluation at all
os.environ['TEST_RESULTS_CSV'] = f'{EXP_DIR}/results_test.csv' # filename of .csv where test metrics will be output by `train.py`
os.environ['LANGS'] = "['english']" # "['chinese', 'english', 'spanish', 'yoruba']" # used to select testfiles; i.e., each fine-tuned model will be tested on each of these languages
os.environ['BULK_TESTING_MODIFIER'] = "-1" # "30" # in `train.py`, adjusts the test_number; useful for multiple training sets e.g. train_31, train_61 that are tested on a single file e.g. test_1 (in which case, use "30"); set to -1 for no bulk testing adjustment

####################################################################
''' (2) RUN FINE-TUNING AND EVALUATION ON EACH EXPERIMENT FILE '''
####################################################################

from manifest import manifest # import the manifest now (now that setup is complete)

logging.basicConfig(format='[%(name)s] [%(levelname)s] %(asctime)s %(message)s')

# ---------- log meta-info ---------- #
f = open(f"{EXP_DIR}/notes.txt", "a")
f.write(str(datetime.now()) + '\n')
f.write(os.getenv('LOG') + '\n')

for name, value in os.environ.items():
    f.write(f"{name}: {value}\n")
f.close() # this is also needed to "flush" output out of buffer

start = time.time()
# ----------------------------------- #

# iterate through each item in the manifest
for item in manifest:
    # define train and test directories
    traindir = os.path.join(EXP_DIR, item['trainfile'])
    testdir = os.path.join(EXP_DIR, item['testfile'])
    model_output_dir = os.path.join(EXP_DIR, 'saved_models', item['model_name'])
    # make model output dir
    os.system(f'mkdir -p {model_output_dir}')

    # make sure train and test files exist
    if not all(os.path.exists(i) for i in [traindir, testdir, model_output_dir]):
        logging.critical('missing dataset file(s) for model', item['model_name'])
    else:
        # define logger
        # Create and configure logger
        logger = logging.getLogger(item['model_name'])
        logger.setLevel(logging.INFO)

        # run trainer
        model = run_trainer(traindir, 
                            testdir, 
                            model_output_dir, 
                            logger)
        del model

# ---------- log meta-info ---------- #
duration = time.time() - start
f = open(f"{EXP_DIR}/notes.txt", "a")
f.write("DURATION: " + str(duration) + '\n\n')
f.close()
# ----------------------------------- #