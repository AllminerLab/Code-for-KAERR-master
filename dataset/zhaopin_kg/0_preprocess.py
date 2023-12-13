"""
Preprocess the data:
1. drop duplicates in table3_action.txt
2. only keep the jd that is satisfied
3. go back to action table, only keep the jd that is satisfied
4. only keep the user that has interaction
"""
import pandas as pd

df_user = pd.read_csv('table1_user.txt', sep='\t')
df_jd = pd.read_csv('table2_jd.txt', sep='\t')
df_action = pd.read_csv('table3_action.txt', sep='\t')

# 只保留成功匹配的jd
df_action.drop_duplicates(keep='first', inplace=True)
df_action.reset_index(drop=True)

valid_jd = set(df_action[df_action['satisfied']==1]['jd_no'].unique().tolist())
df_jd = df_jd[df_jd['jd_no'].isin(valid_jd)].reset_index(drop=True)

# 只保留有效jd的action
df_action = df_action[df_action['jd_no'].isin(valid_jd)].reset_index(drop=True)

# 只保留有交互的user
valid_user = set(df_action['user_id'].unique().tolist())
df_user = df_user[df_user['user_id'].isin(valid_user)].reset_index(drop=True)

# 清洗df_action
valid_user = set(df_user['user_id'].unique().tolist())
valid_jd = set(df_jd['jd_no'].unique().tolist())
df_action = df_action[df_action['user_id'].isin(valid_user) & df_action['jd_no'].isin(valid_jd)].reset_index(drop=True)

print('After preprocessing:')
print('user count: ', df_user.shape[0])
print('jd count: ', df_jd.shape[0])
print('action count: ', df_action.shape[0])

df_user.fillna('', inplace=True)
df_jd.fillna('', inplace=True)
df_action.fillna('', inplace=True)

df_user.to_csv('table1_user_processed.txt', sep='\t', index=False)
df_jd.to_csv('table2_jd_processed.txt', sep='\t', index=False)
df_action.to_csv('table3_action_processed.txt', sep='\t', index=False)