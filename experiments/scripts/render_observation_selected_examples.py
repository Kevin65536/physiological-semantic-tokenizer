#!/usr/bin/env python3
"""Plot final frozen reconstructions for the first registered example per development dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
G=BASE/'20261120_gain_v1'; A=BASE/'20261120_allocation_v1'
DB={'eeg_fnirs_single_trial':'Single-Trial','refed':'REFED','simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual'}


def render(out):
    sources={};colors=('#252F3D','#007E76');bits={}
    for m,p in [('eeg',G),('hb',A)]:
        s=json.loads((p/'summary.json').read_text());q=s['modalities'][m]
        if s['status']!='completed' or not q['met_all_screens']:raise ValueError('completed selected operating point required')
        sources[m]={e['dataset']:e for e in q['examples']};bits[m]=q['selected_bits'];assert set(sources[m])==set(DB)
    if out.exists():raise FileExistsError('preserve original example export')
    with plt.rc_context({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':9}):
        fig,axes=plt.subplots(4,3,figsize=(13.2,8.1),gridspec_kw={'width_ratios':[1.55,1,1]},squeeze=False)
        fig.subplots_adjust(left=.092,right=.985,top=.81,bottom=.118,hspace=.68,wspace=.21)
        fig.text(.03,.952,'What do the selected packets actually reconstruct?',size=21,weight='bold',color='#203A4B');
        fig.text(.03,.912,'First retained examples per dataset, not chosen by reconstruction error. Each pair of traces uses the same frozen training scale.',size=11,color='#3D5464')
        fig.legend([Line2D([0],[0],color=colors[0],lw=1.4),Line2D([0],[0],color=colors[1],lw=1.3,ls='--')],['Observed G0 X','Decoded from packed integers'],loc='upper left',bbox_to_anchor=(.025,.901),ncol=2,frameon=False,fontsize=10)
        for col,ttl in enumerate([f'EEG   |   {bits["eeg"]} bits / 2 s',f'HbO   |   {bits["hb"]} bits / 2 s pair','HbR   |   same paired packet']):
            ax=axes[0,col];ax.text(0,1.35,ttl,transform=ax.transAxes,weight='bold',fontsize=12,color='#203A4B')
        for ri,(db,title) in enumerate(DB.items()):
            for ci in range(3):
                m='eeg' if ci==0 else 'hb';p=G if m=='eeg' else A;r=sources[m][db];c=0 if ci<2 else 1
                with np.load(p/r['file'],allow_pickle=False) as z:
                    x,zp,ids,ind=z['x'],z['pred'],z['pattern'],z['patch'].astype(int);assert x.shape==zp.shape and len(x)==len(ids)==len(ind)
                if m=='eeg':
                    ref=x[0,0]; pred=zp[0,0]; t=np.arange(len(ref))/200.; desc=f"{r['position']}  ·  ID {ids[0]}  ·  crop patch {ind[0]}";xr=(0,2)
                else:
                    ref=np.full(300,np.nan);pred=ref.copy()
                    for i,k in enumerate(ind):ref[20*k:20*k+20]=x[i,c];pred[20*k:20*k+20]=zp[i,c]
                    t=np.arange(300)/10.;desc=f"{r['position']}  ·  {len(ind)}/15 scoreable patches";xr=(0,30)
                ax=axes[ri,ci];ax.plot(t,ref,color=colors[0],lw=1.35,alpha=.9);ax.plot(t,pred,color=colors[1],ls='--',lw=1.22);ax.set_xlim(*xr);ax.margins(y=.12);ax.grid(axis='y',alpha=.14);ax.tick_params(labelsize=8,length=2)
                ax.set_title(desc,fontsize=8,loc='left',color='#516575',pad=3)
                if ci==0: ax.set_ylabel(title,weight='bold',color='#203A4B')
                if ri==3:ax.set_xlabel('Seconds within the 2 s patch' if m=='eeg' else 'Seconds within the 30 s crop',fontsize=9)
        fig.text(.03,.033,'Black and dashed curves share a y-axis within each panel; y-limits differ across panels and sites. Blank segments were not scored. EEG and Hb rows do not establish coupling.',fontsize=9,color='#3D5464')
        fig.savefig(out,dpi=210,facecolor='white');plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True);render(cli.parse_args().output)
