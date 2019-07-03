import os
import pandas as pd
from tqdm import tqdm


RESULT_DIR = './vcc2018_listening_test_scores'

# generate vcc2018_mos for bootstrap estimation
df = pd.read_csv(os.path.join(RESULT_DIR,'vcc2018_evaluation_mos.txt'), header=None)  
df.columns = ['N0','user_ID', 'set_ID', 'sentence_index', 'system_ID', 'system_ID_2', 'SRC', 'TRG', 'TASK', 'ground_truth','MOS', 'SIMILARITY','blank','timestamp','audio_sample', 'N1']
df = df.drop('SIMILARITY', axis=1)
df = df[['user_ID', 'set_ID', 'sentence_index', 'system_ID', 'SRC', 'TRG',
       'TASK', 'MOS', 'audio_sample', 'timestamp']]
df.to_csv('./vcc2018_mos.csv', encoding='utf-8', index=False)

# generate mos_list for training and testing
df = pd.read_csv(os.path.join(RESULT_DIR,'vcc2018_evaluation_mos_simple.txt'),index_col=0)  
df = df.drop('SCORE', axis=1)
df.columns = ['mos']

new_df = pd.DataFrame(columns=['audio', 'mean_mos'])
u_list = df.index.unique()
for i in tqdm(range(len(u_list))):
    f = u_list[i]
    mean_mos = df.loc[df.index == f].mean().values.item()
    new_df = new_df.append({'audio': f, 'mean_mos': mean_mos}, ignore_index=True)

new_df.to_csv('mos_list.txt', sep=',', index=False, header=False)

