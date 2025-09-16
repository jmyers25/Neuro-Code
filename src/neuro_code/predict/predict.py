import glob
import math
import numpy as np
import os
import pandas as pd
import re
from argparse import ArgumentParser

#Usage: python pred_rl.py --model_name Fit_alpha-beta-exp_neg-exp_pos_Fix
#output: /oak/stanford/groups/russpold/users/zenkavi/DevStudy_ServerScripts/fit_rl/.preds/Preds_Fit_{MODEL_NAME}_{SUBNUM}.csv

todo_path = os.environ['TODO_PATH']
server_scripts = os.environ['SERVER_SCRIPTS']
data_loc = os.environ['DATA_LOC']

parser = ArgumentParser()
parser.add_argument("-m", "--model_name", help="model name")
parser.add_argument("-dp", "--data_path", default=todo_path+'/behav_data_tb_organized/machine_game/' , help="data path")
parser.add_argument("-op", "--output_path", default=server_scripts+'/fit_rl/.preds', help="output path")
parser.add_argument("-sr", "--save_by_run", default=False, help="save predicted values broken down by each scanning run")
parser.add_argument("-da", "--data_amt", default=1, help="amount of data to be predicted")
args = parser.parse_args()

model_name = args.model_name
data_path = args.data_path
data_amt = args.data_amt
data_amt_path = data_amt.replace(".", "_")
data_amt = float(data_amt)
if(data_amt == 1):
    output_path = args.output_path
else:
    output_path = args.output_path+ '_'+ data_amt_path
print("Output will be saved in %s"%(output_path))
if not os.path.exists(output_path):
    os.makedirs(output_path)
save_by_run = args.save_by_run

machine_game_data = glob.glob('%s/ProbLearn*'%(data_path))
machine_game_data.sort()

if data_amt == 1:
    model_pars_data = pd.read_csv(server_scripts+'/fit_rl/.fits/LearningParams_'+model_name+'_All.csv')
else:
    fit_data_amt = 1-data_amt
    fit_data_amt = '_'+str(fit_data_amt).replace('.', '_')
    model_pars_data = pd.read_csv(server_scripts+'/fit_rl/.fits%s/LearningParams_'%(fit_data_amt)+model_name+'_All%s.csv'%(fit_data_amt))

#Get best parameters (those with lowest neglogprob) for all subjects
model_pars_data = model_pars_data[model_pars_data['neglogprob'] == model_pars_data.groupby('sub_id')['neglogprob'].transform('min')]
model_pars_data = model_pars_data.sort_values('neglogprob').drop_duplicates('sub_id')

def get_predicted_df(data, pars_dict):
    data = data.reset_index()
    TrialNum = data.Trial_type
    Response = data.Response
    Outcome = data.Points_earned
    EV = [0,0,0,0]
    Prediction_Error = 0
    choiceprob = np.zeros((len(TrialNum)))
    data['EV'] = np.nan
    data['PE'] = np.nan
    data['choiceprob'] = np.nan
    if 'alpha' in pars_dict.keys():
        alpha=pars_dict['alpha']
    else:
        alphaneg=pars_dict['alpha_neg']
        alphapos=pars_dict['alpha_pos']
    beta=pars_dict['beta']
    if 'exp' in pars_dict.keys():
        exp=pars_dict['exp']
    else:
        expneg=pars_dict['exp_neg']
        exppos=pars_dict['exp_pos']
    lossave=pars_dict['lossave']
    for i in range(len(TrialNum)):
        if Response[i] == 0:
            choiceprob[i] = 1
        if Response[i] == 1:
            if EV[int(TrialNum[i]-1)] < 0:
                choiceprob[i] = math.exp(lossave*EV[int(TrialNum[i]-1)]*beta)/(math.exp(lossave*EV[int(TrialNum[i]-1)]*beta)+1)
            else:
                choiceprob[i] = math.exp(EV[int(TrialNum[i]-1)]*beta)/(math.exp(EV[int(TrialNum[i]-1)]*beta)+1)
        if Response[i] == 2:
            if EV[int(TrialNum[i]-1)] < 0:
                choiceprob[i] = 1-math.exp(lossave*EV[int(TrialNum[i]-1)]*beta)/(math.exp(lossave*EV[int(TrialNum[i]-1)]*beta)+1)
            else:
                choiceprob[i] = 1-math.exp(EV[int(TrialNum[i]-1)]*beta)/(math.exp(EV[int(TrialNum[i]-1)]*beta)+1)
        if Outcome[i] != 0:
            if Outcome[i] > EV[int(TrialNum[i]-1)]:
                if 'alpha' in vars() or 'alpha' in globals():
                    if 'exp' in vars() or 'exp' in globals():
                        Prediction_Error = alpha*(Outcome[i] - EV[int(TrialNum[i]-1)])**exp
                    else:
                        Prediction_Error = alpha*(Outcome[i] - EV[int(TrialNum[i]-1)])**exppos
                elif 'exp' in vars() or 'exp' in globals():
                    Prediction_Error = alphapos*(Outcome[i] - EV[int(TrialNum[i]-1)])**exp
                else:
                    Prediction_Error = alphapos*(Outcome[i] - EV[int(TrialNum[i]-1)])**exppos
            if Outcome[i] < EV[int(TrialNum[i]-1)]:
                if 'alpha' in vars() or 'alpha' in globals():
                    if 'exp' in vars() or 'exp' in globals():
                        Prediction_Error = -1*alpha*(EV[int(TrialNum[i]-1)]-Outcome[i])**exp
                    else:
                        Prediction_Error = -1*alpha*(EV[int(TrialNum[i]-1)]-Outcome[i])**expneg
                elif 'exp' in vars() or 'exp' in globals():
                    Prediction_Error = -1*alphaneg*(EV[int(TrialNum[i]-1)]-Outcome[i])**exp
                else:
                    Prediction_Error = -1*alphaneg*(EV[int(TrialNum[i]-1)]-Outcome[i])**expneg
            if Outcome[i] == EV[int(TrialNum[i]-1)]:
                Prediction_Error = 0
            data.EV[i] = EV[int(TrialNum[i]-1)]
            data.PE[i] = Prediction_Error
            data.choiceprob[i] = choiceprob[i]
            EV[int(TrialNum[i]-1)] += Prediction_Error
        elif Outcome[i] == 0:
            data.EV[i] = EV[int(TrialNum[i]-1)]
            data.choiceprob[i] = choiceprob[i]
    return(data)

for subject_data in machine_game_data:

    df = pd.read_csv(subject_data)
    nrows = (-1)*round(df.shape[0] * data_amt)
    df = df[nrows:]
    print("Making predictions for %s of trials"%(data_amt))

    subnum = re.findall('\d+', os.path.basename(subject_data))[0]
    print('Beginning model predictions for sub-%s'%(subnum))

    pars_row = model_pars_data.query("sub_id == %s"%(float(subnum)))

    if pars_row.shape[0]>0:
        pars_dict = pars_row.filter(regex='xopt').to_dict('records')[0]
        pars_dict = {x.replace('xopt_', ''): v for x, v in pars_dict.items()}

        for (k,v) in pars_dict.items():
            if np.isnan(v):
                fix_row = model_pars_data.query("sub_id == %s"%(float(subnum)))
                fix_dict = fix_row.filter(regex='x0').to_dict('records')[0]
                fix_dict = {x.replace('x0_', ''): v for x, v in fix_dict.items()}
                pars_dict[k] = fix_dict[k]

        df = get_predicted_df(data=df, pars_dict=pars_dict)
        df = df[['Trial_type', 'Response', 'Points_earned', 'EV', 'PE', 'choiceprob']]
        df['sub_id'] = subnum

        if float(data_amt) == 1:
            df.to_csv('%s/Preds_%s_%s.csv'%(output_path,model_name, subnum))
        else:
            df.to_csv('%s/Preds_%s_%s_%s.csv'%(output_path,model_name, subnum, data_amt_path))

        if(save_by_run):
            file_length = df.shape[0]
            N=30
            for i in range(file_length//N):
                run_rows = df[N*(i+1)-(N):N*(i+1)]
                try:
                    if not os.path.exists(os.path.join(output_path, 'sub-%s'%(subnum))):
                        os.makedirs(os.path.join(output_path, 'sub-%s'%(subnum)))
                    run_rows.to_csv(os.path.join(output_path, 'sub-%s'%(subnum),'sub-'+str(subnum)+'_task-machinegame_run-00'+str(i+1)+'_ev_rpe.csv'))
                    print('Done with sub-%s run-%s'%(subnum, str(i+1)))
                except:
                    print('Data not saved for sub-%s run-%s'%(subnum, str(i+1)))
    else:
        print('No model parameters for sub-%s model: %s'%(subnum, model_name))
