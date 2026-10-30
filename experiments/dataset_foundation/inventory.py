"""Summarize scenario/route/driver inventory and split-level sustained controls."""
from pathlib import Path
import json,os,re
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 rare_path=HERE/'results/full_release_rare_coverage.csv'
 if rare_path.exists():
  selection=json.loads((HERE.parents[1]/'data/person2drive_train/manifests/selection.json').read_text());chosen={(e['driver'],e['episode']) for e in selection['selected']}
  rare=pd.read_csv(rare_path);rare['selected']=[(d,e) in chosen for d,e in zip(rare.driver,rare.episode)];rare.to_csv(rare_path,index=False)
 df=pd.read_csv(HERE/'results/episode_coverage.csv');split=json.loads((HERE/'results/splits.json').read_text())['episodes'];assignment={ep:part for part,keys in split.items() for ep in keys}
 df['split']=df.episode.map(assignment);df['scenario_type']=df.episode.map(lambda e:e.split('/')[1].split('_Town')[0] if e.startswith('bench2drive') else 'human_fixed_route')
 df['town']=df.episode.map(lambda e:re.search(r'Town\d+(?:HD)?',e)[0]);df['weather']=df.episode.map(lambda e:re.search(r'Weather(\d+)',e)[1] if 'Weather' in e else '')
 df.to_csv(HERE/'results/scenario_inventory.csv',index=False)
 fields=['frames','one_sec_coast_H0','one_sec_coast_H1','one_sec_partial_H0','one_sec_partial_H1','multi_transition_windows_H0','multi_longitudinal_regime_windows_H0','simultaneous_pedal_frames']
 df.groupby(['source','split'])[fields].sum().to_csv(HERE/'results/split_sequence_coverage.csv')
 fig,axes=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.4,1]})
 bench=df[df.source=='bench2drive'].groupby('scenario_type').frames.sum().sort_values();axes[0].barh(bench.index,bench.values);axes[0].tick_params(axis='y',labelsize=6);axes[0].set(title='Bench2Drive: 44 scenario types',xlabel='Indexed frames')
 person=df[df.source=='person2drive'].pivot_table(index='driver',columns='route',values='frames',aggfunc='sum',fill_value=0)
 im=axes[1].imshow(person.values,aspect='auto',cmap='Blues');axes[1].set_xticks(np.arange(len(person.columns)),person.columns);axes[1].set_yticks(np.arange(len(person)),person.index);axes[1].set(title='Person2Drive: repeated physical routes',xlabel='Physical route')
 for i in range(len(person)):
  for j in range(len(person.columns)):axes[1].text(j,i,str(person.iloc[i,j]),ha='center',va='center',fontsize=7,color='black' if person.iloc[i,j]<person.values.max()*.65 else 'white')
 fig.colorbar(im,ax=axes[1],label='Indexed frames');fig.tight_layout();fig.savefig(HERE/'results/scenario_coverage.png',dpi=160);plt.close(fig)
 print('Scenario and split coverage inventory written',flush=True)
if __name__=='__main__':main()
