import os
import pandas as pd
from tqdm import tqdm
import random
import argparse
import numpy as np
import scipy.stats
from math import sqrt
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

parser = argparse.ArgumentParser()
parser.add_argument("--num_user", type=int, default=134, help="num user to select from in each replication")
parser.add_argument("--iter", type=int, default=1000, help="number iterations")

SEED = 1984
random.seed(SEED)


# get index of key in given array
def getIndex(array, key):
    return int(np.argwhere(array==key))


def getGIndex(array, key):
    index = getIndex(array, key)
    
    return int(np.floor(index/10))

def getGIndexFromRow(array, row):
    key = row['audio_smaple']
    return getGIndex(array, key)
    


if __name__ == '__main__':
    
    args = parser.parse_args()
        
    NUM_USER = args.num_user
    BOOTSTRAP_ITER = args.iter

    # load vcc2018_mos
    df = pd.read_csv('./vcc2018_mos.csv', encoding='utf-8') 

    # df.iloc[df.loc[df['audio_sample'].str.contains('NAT')].index, df.columns.get_loc('MOS')] = 5

    # remove NAT audio
    df = df[df['audio_sample'].str.contains('NAT')==False]

    mos_df = df[['audio_sample', 'MOS']].groupby(['audio_sample']).mean()
    sys_df = df[['system_ID', 'MOS']].groupby(['system_ID']).mean()
    user_list = df['user_ID'].unique()


    print('bootstrap estimation for intrinsic MOS of VCC2018 submitted audio')
    print('number user: {}, and number iterations {}'.format(NUM_USER, BOOTSTRAP_ITER))


    MSEs = []
    MAEs = []
    RMSEs = []
    LCCs = []
    SRCCs = []

    tenMSEs = []
    tenMAEs = []
    tenRMSEs = []
    tenLCCs = []
    tenSRCCs = []

    sysMSEs = []
    sysMAEs = []
    sysRMSEs = []
    sysLCCs = []
    sysSRCCs = []

    # start bootstraping

    for b in tqdm(range(BOOTSTRAP_ITER)):

        # get random sampled users
        random_user = random.sample(list(user_list), NUM_USER)

        # get sub df
        sub_df = df[df['user_ID'].isin(random_user)]

        # for 10 utterance
        # get unique audio list
        u_audio_list = sub_df.audio_sample.unique()

        # get unique_df audio from df
        u_df = df[df['audio_sample'].isin(u_audio_list)]    

        # clustering 
        random.shuffle(u_audio_list)
        group_df = pd.DataFrame(data={'audio_sample': u_audio_list})
        group_df['audio_group'] = np.floor(group_df.index/10) 

        # merge group_df into sub_df and u_df
        sub_df = pd.merge(sub_df, group_df, how='left', on=['audio_sample'])
        u_df = pd.merge(u_df, group_df, how='left', on=['audio_sample'])   
        g_mos_df = u_df[['audio_group', 'MOS']].groupby(['audio_group']).mean()

        # calculate mean
        sub_mos = sub_df[['audio_sample', 'MOS']].groupby(['audio_sample']).mean()
        sub_tenmos = sub_df[['audio_group', 'MOS']].groupby(['audio_group']).mean()
        sub_sys = sub_df[['system_ID', 'MOS']].groupby(['system_ID']).mean()


        # merge selected df with whole df
        merge_mos = pd.merge(sub_mos, mos_df, how='inner', on='audio_sample')
        merge_tenmos = pd.merge(sub_tenmos, g_mos_df, how='inner', on='audio_group')
        merge_sys = pd.merge(sub_sys, sys_df, how='inner', on='system_ID')

        # get two mos list
        mos1 = merge_mos.iloc[:,0].values
        mos2 = merge_mos.iloc[:,1].values

        # get two mos list
        tenmos1 = merge_tenmos.iloc[:,0].values
        tenmos2 = merge_tenmos.iloc[:,1].values

        sys1 = merge_sys.iloc[:,0].values
        sys2 = merge_sys.iloc[:,1].values


        # calculate statistics for utterance, MSE, RMSE, MAE, rho, rho_s
        mse = MSE(mos1, mos2)
        rmse = sqrt(mse)
        mae = MAE(mos1, mos2)
        lcc = scipy.stats.pearsonr(mos1, mos2)[0]
        srcc = scipy.stats.spearmanr(mos1, mos2)[0]

        # add to list
        MSEs.append(mse)
        RMSEs.append(rmse)
        MAEs.append(mae)
        LCCs.append(lcc)
        SRCCs.append(srcc)  


        # calculate statistics for 10utterance, MSE, RMSE, MAE, rho, rho_s
        tenmse = MSE(tenmos1, tenmos2)
        tenrmse = sqrt(tenmse)
        tenmae = MAE(tenmos1, tenmos2)
        tenlcc = scipy.stats.pearsonr(tenmos1, tenmos2)[0]
        tensrcc = scipy.stats.spearmanr(tenmos1, tenmos2)[0]

        # add to list
        tenMSEs.append(tenmse)
        tenRMSEs.append(tenrmse)
        tenMAEs.append(tenmae)
        tenLCCs.append(tenlcc)
        tenSRCCs.append(tensrcc)  



        # system level correlation
        # calculate statistics, MSE, RMSE, MAE, rho, rho_s
        smse = MSE(sys1, sys2)
        srmse = sqrt(smse)
        smae = MAE(sys1, sys2)
        slcc = scipy.stats.pearsonr(sys1, sys2)[0]
        ssrcc = scipy.stats.spearmanr(sys1, sys2)[0]

        # add to list
        sysMSEs.append(smse)
        sysRMSEs.append(srmse)
        sysMAEs.append(smae)
        sysLCCs.append(slcc)
        sysSRCCs.append(ssrcc)  

    MSEs = np.array(MSEs)
    RMSEs = np.array(RMSEs)
    MAEs = np.array(MAEs)
    LCCs = np.array(LCCs)
    SRCCs = np.array(SRCCs)

    tenMSEs = np.array(tenMSEs)
    tenRMSEs = np.array(tenRMSEs)
    tenMAEs = np.array(tenMAEs)
    tenLCCs = np.array(tenLCCs)
    tenSRCCs = np.array(tenSRCCs)

    sysMSEs = np.array(sysMSEs)
    sysRMSEs = np.array(sysRMSEs)
    sysMAEs = np.array(sysMAEs)
    sysLCCs = np.array(sysLCCs)
    sysSRCCs = np.array(sysSRCCs)


    print('===========================================')
    print('============== utterance level ============')
    print('===========================================')
    print('\n\t\tMEAN\tSD\tMIN\tMAX')
    print('\tMSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(MSEs.mean(), MSEs.std(), MSEs.min(), MSEs.max()))
    print('\tRMSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(RMSEs.mean(), RMSEs.std(), RMSEs.min(), RMSEs.max()))
    print('\tMAE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(MAEs.mean(), MAEs.std(), MAEs.min(), MAEs.max()))
    print('\tLCC\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(LCCs.mean(), LCCs.std(), LCCs.min(), LCCs.max()))
    print('\tSRCC\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(SRCCs.mean(), SRCCs.std(), SRCCs.min(), SRCCs.max()))
    print('')
    print('===========================================')
    print('============== 10 utterance level =========')
    print('===========================================')
    print('\n\t\tMEAN\tSD\tMIN\tMAX')
    print('\tMSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(tenMSEs.mean(), tenMSEs.std(), tenMSEs.min(), tenMSEs.max()))
    print('\tRMSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(tenRMSEs.mean(), tenRMSEs.std(), tenRMSEs.min(), tenRMSEs.max()))
    print('\tMAE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(tenMAEs.mean(), tenMAEs.std(), tenMAEs.min(), tenMAEs.max()))
    print('\tLCC\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(tenLCCs.mean(), tenLCCs.std(), tenLCCs.min(), tenLCCs.max()))
    print('\tSRCC\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(tenSRCCs.mean(), tenSRCCs.std(), tenSRCCs.min(), tenSRCCs.max()))
    print('')
    print('===========================================')
    print('============== system level ===============')
    print('===========================================')
    print('\n\t\tMEAN\tSD\tMIN\tMAX')
    print('\tMSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(sysMSEs.mean(), sysMSEs.std(), sysMSEs.min(), sysMSEs.max()))
    print('\tRMSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(sysRMSEs.mean(), sysRMSEs.std(), sysRMSEs.min(), sysRMSEs.max()))
    print('\tMAE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(sysMAEs.mean(), sysMAEs.std(), sysMAEs.min(), sysMAEs.max()))
    print('\tLCC\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(sysLCCs.mean(), sysLCCs.std(), sysLCCs.min(), sysLCCs.max()))
    print('\tSRCC\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(sysSRCCs.mean(), sysSRCCs.std(), sysSRCCs.min(), sysSRCCs.max()))

