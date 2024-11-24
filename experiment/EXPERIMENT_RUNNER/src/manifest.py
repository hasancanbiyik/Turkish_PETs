import os

manifest = []
start_filenum = os.getenv('START_FILENUM')
end_filenum = os.getenv('END_FILENUM')
exp_dir = os.getenv('EXP_DIR')

for x in range(int(start_filenum), int(end_filenum)):
    num = str(x)
    test_obj = {'model_name': 'finetuned_'+num,
     'trainfile': 'train_{}.csv'.format(num),
     'testfile': 'val_{}.csv'.format(num)}
    manifest.append(test_obj)