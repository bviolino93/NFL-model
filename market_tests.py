"""
Do schedule, weather and injury facts predict results BEYOND the closing line?

Each test regresses (result - close) on a feature, with robust standard
errors, separately on an early and a late half of the sample. A feature is
only worth betting if its coefficient has the same sign and |t| > 2 in BOTH
halves; with ~20 features tested, one |t| > 2 in one half is expected by
chance.

Findings when written (2026-10-25):
  wind (observed mph, outdoor)  totals  -0.15/mph  t -2.5 and -2.9  <- real,
      but nflverse wind is OBSERVED at kickoff, not forecast, so it overstates
      what was knowable when betting. Re-test with archived forecasts.
  injury loads (snap-weighted, non-QB)  move margins ~1 pt per starter, and
      the close already prices ~85% of it; residual flips sign by half.
  rest, byes, division, QB change, weeks 17-18: nothing consistent.

    python market_tests.py
"""
import os
import re

import numpy as np
import pandas as pd

from nfl_data import TEAM_CANON, graded_games, load_games


def schedule_tests():
    g=graded_games(load_games()); g=g[g.season>=2006].copy()
    g=g.sort_values(['season','week'])
    # previous starter per team
    long=pd.concat([g[['game_id','season','week','home_team','home_qb_id']].rename(columns={'home_team':'t','home_qb_id':'qb'}),
                    g[['game_id','season','week','away_team','away_qb_id']].rename(columns={'away_team':'t','away_qb_id':'qb'})]).sort_values(['season','week'])
    long['prev']=long.groupby('t')['qb'].shift(1)
    long['qbchg']=((long.qb!=long.prev)&long.prev.notna()).astype(float)
    ch=long.set_index(['game_id','t'])['qbchg']
    g['h_qbchg']=[ch.get((i,t),0) for i,t in zip(g.game_id,g.home_team)]
    g['a_qbchg']=[ch.get((i,t),0) for i,t in zip(g.game_id,g.away_team)]
    g['rest_diff']=(g.home_rest-g.away_rest).clip(-7,7)
    g['h_bye']=(g.home_rest>=13).astype(float); g['a_bye']=(g.away_rest>=13).astype(float)
    g['short']=((g.home_rest<=5)&(g.away_rest<=5)).astype(float)
    g['outdoor']=g.roof.isin(['outdoors','open']).astype(float)
    g['wind']=pd.to_numeric(g.wind,errors='coerce').fillna(0)*g.outdoor
    g['wind15']=(g.wind>=15).astype(float)
    g['cold']=((pd.to_numeric(g.temp,errors='coerce')<=32)&(g.outdoor==1)).astype(float)
    g['div']=g.div_game.astype(float)
    g['wk18']=((g.week>=17)&(g.game_type=='REG')).astype(float)
    g['fav_home']=(g.spread_line>0).astype(float)
    g['sres']=g.margin-g.spread_line; g['tres']=g.points-g.total_line
    def ols(df,y,xs):
        X=np.column_stack([np.ones(len(df))]+[df[x].to_numpy(float) for x in xs]); Y=df[y].to_numpy(float)
        b,*_=np.linalg.lstsq(X,Y,rcond=None); r=Y-X@b
        XtXi=np.linalg.pinv(X.T@X); meat=(X*r[:,None]).T@(X*r[:,None]); V=XtXi@meat@XtXi
        return pd.DataFrame({'b':b,'t':b/np.sqrt(np.diag(V))},index=['const']+xs).round(3)
    sx=['rest_diff','h_bye','a_bye','h_qbchg','a_qbchg','div','wk18']
    tx=['wind','wind15','cold','short','div','wk18','h_qbchg','a_qbchg']
    for lab,d in [('2006-2015',g[g.season<=2015]),('2016-2026',g[g.season>=2016])]:
        print(lab,len(d)); print(pd.concat([ols(d,'sres',sx).add_prefix('spr_'),],axis=1).T.to_string())
        print(ols(d,'tres',tx).T.to_string())
    print('wind games share',g.wind15.mean().round(3), 'outdoor wind missing for 2026?', g[g.season==2026].wind.describe().round(1).to_dict())


def injury_tests():
    C = os.environ.get('NFL_CACHE_DIR', '.cache_nflverse')
    def nk(s): s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?","",str(s).lower()); return re.sub(r"[^a-z]","",s)
    canon=lambda s: s.astype(str).replace({**TEAM_CANON,'LAR':'LA','JAC':'JAX'})
    yrs=range(2013,2027)
    sn=pd.concat([pd.read_parquet(f'{C}/snap_counts_{y}.parquet') for y in yrs])
    sn['team']=canon(sn.team); sn['k']=sn.player.map(nk)
    sn['pct']=np.maximum(sn.offense_pct.fillna(0),sn.defense_pct.fillna(0))
    sn['side']=np.where(sn.offense_pct.fillna(0)>=sn.defense_pct.fillna(0),'off','def')
    inj=pd.concat([pd.read_parquet(f'{C}/injuries_{y}.parquet') for y in yrs])
    inj['team']=canon(inj.team); inj['k']=inj.full_name.map(nk)
    P={'Out':1.0,'Doubtful':0.9,'Questionable':0.25}
    inj['p']=inj.report_status.map(P).fillna(0)
    inj=inj[(inj.p>0)&(inj.position!='QB')]
    # importance = mean pct over team's previous up to 4 games this season (0 if absent)
    tw=sn[['season','week','team']].drop_duplicates()
    rows=[]
    for (s,t),grp in sn.groupby(['season','team']):
        weeks=sorted(grp.week.unique())
        piv=grp.pivot_table(index='week',columns='k',values='pct',aggfunc='max').reindex(weeks).fillna(0)
        side=grp.groupby('k').side.agg(lambda x:x.mode()[0])
        imp=piv.rolling(4,min_periods=1).mean().shift(1)  # before week
        # map to the NEXT game week: imp at index week w = mean of games before w
        for w in weeks[1:]:
            r=imp.loc[w]; r=r[r>0.25]
            for k,v in r.items(): rows.append((s,w,t,k,v,side[k]))
    imp=pd.DataFrame(rows,columns=['season','week','team','k','imp','side'])
    m=inj.merge(imp,on=['season','week','team','k'])
    m['load']=m.p*m.imp
    L=m.pivot_table(index=['season','week','team'],columns='side',values='load',aggfunc='sum').fillna(0).reset_index()
    g=graded_games(load_games()); g=g[(g.season>=2013)&(g.week>=2)]
    for side_t,pre in (('home_team','h'),('away_team','a')):
        g=g.merge(L.rename(columns={'team':side_t,'off':pre+'_off','def':pre+'_def'}),on=['season','week',side_t],how='left')
    for c in ['h_off','h_def','a_off','a_def']: g[c]=g[c].fillna(0)
    g['sres']=g.margin-g.spread_line; g['tres']=g.points-g.total_line
    g['d_off']=g.h_off-g.a_off; g['d_def']=g.h_def-g.a_def
    g['s_off']=g.h_off+g.a_off; g['s_def']=g.h_def+g.a_def
    print('mean loads',g[['h_off','h_def']].mean().round(2).to_dict(), 'n',len(g))
    def ols(df,y,xs):
        X=np.column_stack([np.ones(len(df))]+[df[x].to_numpy(float) for x in xs]); Y=df[y].to_numpy(float)
        b,*_=np.linalg.lstsq(X,Y,rcond=None); r=Y-X@b
        XtXi=np.linalg.pinv(X.T@X); V=XtXi@((X*r[:,None]).T@(X*r[:,None]))@XtXi
        return pd.DataFrame({'b':b,'t':b/np.sqrt(np.diag(V))},index=['c']+xs).round(3).T
    for lab,d in [('2013-2019',g[g.season<=2019]),('2020-2026',g[g.season>=2020])]:
        print(lab,len(d)); print(ols(d,'sres',['d_off','d_def']).to_string()); print(ols(d,'tres',['s_off','s_def']).to_string())
    # raw (no market): does load move margins at all?
    print('raw margin', ols(g,'margin',['d_off','d_def']).to_string())
    print('market absorbs?', ols(g,'spread_line',['d_off','d_def']).to_string())


if __name__ == '__main__':
    schedule_tests()
    injury_tests()
