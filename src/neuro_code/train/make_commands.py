import csv
import numpy as np
import pandas as pd
import os

todo_path = os.environ['TODO_PATH']
server_scripts = os.environ['SERVER_SCRIPTS']

beh_file_names = os.listdir(todo_path+"/behav_data_tb_organized/machine_game")

subjects = [ x.split("ProbLearn")[1].split(".")[0] for x in beh_file_names ]

data_path = todo_path+"/behav_data_tb_organized/machine_game/"

output_path = server_scripts+"/fit_rl/.fits/"

tasklist_path = server_scripts+"/fit_rl/.rl_task_lists/"

n_fits = 50

rl_models = pd.read_csv('possible_rl_models.csv')

pars_list = []

for i in range(len(rl_models)):

    #cols where row is nan
    #rl_models.columns[rl_models.iloc[0].isnull()]

    fit_pars = list(rl_models.columns[(rl_models == "1").iloc[i]])
    pars = dict(zip(fit_pars, len(fit_pars)*[np.nan]))

    #What to fix these too if they are fixed??
    #Fix values:
    #alpha - 0.05
    #beta - 1
    #exp - 0.5
    fix_pars = list(rl_models.columns[(rl_models == "fix").iloc[i]])

    for par in fix_pars:
        if 'alpha' in par:
            pars[par] = 0.05
        elif 'exp' in par:
            pars[par] = 1
        elif 'beta' in par:
            pars[par] = 1


    pars_list.append(pars)

def extract_pars(pars):
    fixparams = []
    fitparams = []
    for key in sorted(pars.keys()):
        if np.isnan(pars[key]):
            fitparams.append(key)
        else:
            fixparams.append(key)
    out = {'fitparams':fitparams, 'fixparams':fixparams}
    return(out)

task_list = []

for pars in pars_list:
    for subject in subjects:
        command = 'python fit_rl.py --subject ' + str(subject) + ' --pars "' + str(pars) + '"'
        #python  fit_rl.py  --subject 100188 --pars "{'exp':  1 ,  'alpha':  nan ,  'beta':  nan}"
        task_list.append(command)
        fitparams = '-'.join(extract_pars(pars)['fitparams'])
        fixparams = '-'.join(extract_pars(pars)['fixparams'])
        if len(fixparams) == 0:
            file_name = 'fit_'+ fitparams + '_fix'+ fixparams +'_task_list.sh'
        else:
            file_name = 'fit_'+ fitparams + '_fix_'+ fixparams +'_task_list.sh'
    pd.DataFrame(task_list).to_csv(tasklist_path+file_name, header=False, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
    task_list = []

#RUN THESE TWO ON CMD LINE in .rl_task_lists
#find ./ -type f -exec sed -i -e 's/{/"{/g' {} \;
#find ./ -type f -exec sed -i -e 's/}/}"/g' {} \;

#Sample command
#python fit_rl.py {SUBJECT} {N_FITS} {DATA_PATH} {OUTPUT_PATH} {PARS}
