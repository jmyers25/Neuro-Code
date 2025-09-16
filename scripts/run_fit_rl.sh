set -e
for task_list in fit_alpha_neg-alpha_pos-beta-exp_neg-exp_pos-lossave_fix_task_list.sh fit_alpha_neg-alpha_pos-beta-exp_fix_lossave_task_list.sh fit_alpha-beta-lossave_fix_exp_task_list.sh fit_alpha_neg-alpha_pos-beta-exp_neg-exp_pos_fix_lossave_task_list.sh fit_alpha-beta-exp-lossave_fix_task_list.sh fit_alpha-beta_fix_exp-lossave_task_list.sh fit_alpha-beta-exp_neg-exp_pos-lossave_fix_task_list.sh fit_alpha-beta-exp_fix_lossave_task_list.sh fit_alpha_neg-alpha_pos-beta-exp-lossave_fix_task_list.sh fit_alpha-beta-exp_neg-exp_pos_fix_lossave_task_list.sh fit_alpha_neg-alpha_pos-beta-lossave_fix_exp_task_list.sh fit_alpha_neg-alpha_pos-beta_fix_exp-lossave_task_list.sh
do
sed -e "s/{TASK_LIST}/$task_list/g" fit_rl.batch | sbatch
done
