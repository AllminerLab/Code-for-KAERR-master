""""
使用TechGPT从岗位要求中提取关键词
"""
import pandas as pd
import os
import argparse
from tqdm import tqdm
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, default='table2_jd_processed.txt')
parser.add_argument('-o', '--output_file', type=str, default='table2_jd_processed_gpt.txt')
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-s', '--start', type=int, default=0)
parser.add_argument('-e', '--end', type=int, default=-1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from tech_gpt import infer

input_file = args.input_file
df_jd  = pd.read_csv(input_file, sep="\t")
df_jd = df_jd.fillna('')

# 对重复的岗位描述进行标记
# 后续出现的job_description改为该描述第一次出现的index
jd2index = {}
for index, row in tqdm(df_jd.iterrows(), total=df_jd.shape[0]):
    if row['job_description'] in jd2index:
        df_jd.loc[index, 'job_description'] = str(jd2index[row['job_description']])
    else:
        jd2index[row['job_description']] = index

if args.end == -1:
    df_jd = df_jd[args.start:]
else:
    df_jd = df_jd[args.start:args.end]

def keyword_extract(desc):
    if len(desc) < 10:
        return ''
    prompt = f"Human: 提取下面这段岗位要求文本中与技能要求相关的关键词，用关键词回答，关键词之间用|隔开。\n岗位职责1、负责公司行政、后勤管理工作；2、负责公司公文档案管理；3、负责办公区域的日常管理，维护办公区正常办公秩序；4、负责公司会议管理，年中、年终各类会议及晚会的筹备、组织；5、行政费用管控，编制年度行政费用预算，月度滚动控制；6、负责公司商务、政府接待，后勤保障；7、负责公司物料采购、固定资产管理等；8、完成公司领导交办的其他工作。任职要求：1、行政管理、企业管理等相关专业本科及以上学历；2、具备5年以上行政工作经验，3年以上部门管理工作经验；3、具备良好的企业行政管理知识，熟悉国家相关行政法律法规；4、具备良好的沟通协调和管理能力。 \n\nAssistant: 行政管理|后勤管理|公文档案管理|办公区域管理|会议管理|行政费用管控|商务接待|政府接待|后勤保障|物资采购|固定资产管理 \n\nHuman: 继续提取下面这段文本。{desc} \n\nAssistant: "
    answer = infer(prompt)
    return answer.split('Assistant: ')[-1].replace('\n', ' ')

df_jd['skill_keyword'] = df_jd['job_description'].progress_apply(keyword_extract)

df_jd.to_csv(args.output_file, sep="\t", index=False)

