"""导出可视化页面用的回放数据：python -m ormat.export_replay > replay_data.json"""
import json, csv, glob, os, sys, numpy as np, torch
from ormat.envs import EnvConfig, ReplenishmentEnv
from ormat.mat import MultiAgentTransformer
from ormat.baselines import BaseStockPolicy, PriorityBaseStockPolicy, critical_ratio_levels
R=os.path.dirname(os.path.abspath(__file__))+"/"
cfg=EnvConfig(capacity_ratio=0.85)
S_cr=np.array(critical_ratio_levels(cfg),float)
cap=ReplenishmentEnv(cfg).capacity
def mat(mode,seed):
    env=ReplenishmentEnv(cfg)
    m=MultiAgentTransformer(env.obs_dim,env.n_actions,4,order_mode=mode)
    m.load_state_dict(torch.load(f"{R}results_cap0.85/{mode}_s{seed}/model.pt")); m.eval(); return m
policies=[
 ("learned","MAT 学习顺序","mat",mat("learned",2)),
 ("fixed","MAT 固定顺序","mat",mat("fixed",0)),
 ("priority","按优先级配给","bl",PriorityBaseStockPolicy(np.maximum(S_cr-2,0),cfg.stockout_cost,cap,max_order=12)),
 ("sS","(s,S) 网格搜索","bl",BaseStockPolicy(S_cr-3,S_cr-7,12)),
]
episodes=[]
for key,name,kind,pol in policies:
    env=ReplenishmentEnv(cfg,n_envs=1,seed=2026); obs=env.reset(); steps=[]; done=False
    while not done:
        order=None
        if kind=="mat":
            a,_,_,perm,_=pol.act(torch.as_tensor(obs),deterministic=True); a=a.numpy(); order=perm[0].tolist()
        else: a=pol.act(env)
        obs,_,done,info=env.step(a)
        steps.append(dict(req=a[0].astype(int).tolist(), fill=env.last_filled[0].astype(int).tolist(),
            dem=env.last_demand[0].astype(int).tolist(), inv=env.inventory[0].astype(int).tolist(),
            pipe=env.pipeline[0].sum(-1).astype(int).tolist(), lost=env.last_lost[0].astype(int).tolist(),
            cost=round(float(info["cost"][0]),1), order=order))
    episodes.append(dict(key=key,name=name,steps=steps,total=round(sum(s["cost"] for s in steps),1)))
    print(name, episodes[-1]["total"], file=sys.stderr)
curves={}
for cap_dir,label in (("results","1.05"),("results_cap0.85","0.85")):
    modes={}
    for path in sorted(glob.glob(f"{R}{cap_dir}/*_s*/progress.csv")):
        mode=os.path.basename(os.path.dirname(path)).rsplit("_s",1)[0]
        rows=list(csv.DictReader(open(path)))
        modes.setdefault(mode,[]).append([float(r["eval_cost"]) for r in rows])
        steps=[int(r["env_steps"]) for r in rows]
    curves[label]=dict(steps=steps, modes={m:dict(mean=np.mean(v,0).round(1).tolist(), std=np.std(v,0).round(1).tolist()) for m,v in modes.items()})
summary={"1.05":{"base-stock":1144.0,"(s,S)":1115.8,"priority":1123.1,"fixed":[1120.1,10.1],"random":[1133.0,8.8],"learned":[1132.8,6.1],"none":[1128.7,3.4]},
         "0.85":{"base-stock":2516.2,"(s,S)":1594.1,"priority":1272.2,"fixed":[1388.7,15.5],"random":[1386.9,13.4],"learned":[1362.0,19.4],"none":[1375.5,13.5]},
         "ranks0.85":{"learned":[0.39,1.32,2.08,2.21]}}
data=dict(cfg=dict(mu=list(cfg.demand_mean),b=list(cfg.stockout_cost),h=list(cfg.holding_cost),cap=cap,horizon=cfg.horizon,lead=cfg.lead_time),episodes=episodes,curves=curves,summary=summary)
json.dump(data,sys.stdout,ensure_ascii=False,separators=(",",":"))
