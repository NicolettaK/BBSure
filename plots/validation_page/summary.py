"""Summary numbers of the final run for the page (same metrics as the October deck)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables import tree_interval


def tree_dof(root, s, model):
    ev = json.load(open(os.path.join(root, 'runs', s, 'tree', 'decision_log.json')))
    f = [e for e in ev if e['node'] == 'fit' and e.get('model') == model and e.get('dof')]
    return f[-1]['dof'] if f else np.nan


def summary(table, root):
    d = pd.read_csv(table)
    p = pd.read_csv(os.path.join(root, 'profile_results.csv'))
    d = d.merge(p[['index', 'kind', 'chi2_at_rstar', 'dof_rstar']], on=['index', 'kind'])
    t = pd.read_csv(os.path.join(root, 'results_unblinded.csv'))[['index', 'kind', 'chi2']]
    d = d.merge(t, on=['index', 'kind'])
    for i, r in d.iterrows():
        if not np.isfinite(r.t_lo):
            d.loc[i, 't_lo'], d.loc[i, 't_hi'] = tree_interval(root, r.set, r.t_r)
    d['t_dof'] = [tree_dof(root, r.set, r.t_model) for r in d.itertuples()]
    out = {'groups': {}}
    for g in ('d9s4', 'rand'):
        v = d[d.kind == g]
        row = {'n': len(v)}
        for k in ('t', 'p'):
            pull = v[f'{k}_pull']; det = v[f'{k}_concl'] == 'r_detection'; ul = ~det
            low = v.r_true < 3e-3; hi = v.r_true >= 0.01
            chi2 = v.chi2 if k == 't' else v.chi2_at_rstar
            dof = v.t_dof if k == 't' else v.dof_rstar
            pte = v[f'{k}_pte']
            row[k] = dict(
                bias=round(pull.mean(), 2), bias_err=round(pull.std(ddof=1) / np.sqrt(len(v)), 2),
                width=round(pull.std(ddof=1), 2),
                in68=int(((v.r_true >= v[f'{k}_lo']) & (v.r_true <= v[f'{k}_hi'])).sum()),
                false_det=int((det & low).sum()), n_low=int(low.sum()),
                det_r0=int((det & (v.r_true == 0)).sum()), n_r0=int((v.r_true == 0).sum()),
                ul_ok=int((ul & (v.r_true <= v[f'{k}_r95'])).sum()), n_ul=int(ul.sum()),
                det_hi=int((det & hi).sum()), n_hi=int(hi.sum()), n_det=int(det.sum()),
                pte_bins=[int(x) for x in np.histogram(pte, [0, .01, .05, .2, .6, 1.0001])[0]],
                chi2_red=round(float((chi2 / dof).mean()), 3))
        row['model_changed'] = int((v.t_model != v.p_model).sum())
        row['concl_changed'] = int((v.t_concl != v.p_concl).sum())
        out['groups'][g] = row
    # paired difference (same CMB + noise): (r_rand - r_d9s4) / sigma_rand
    a = d[d.kind == 'rand'].set_index('index'); b = d[d.kind == 'd9s4'].set_index('index')
    a = a.loc[a.index.intersection(b.index)]; b = b.loc[a.index]
    pair = {}
    for k in ('t', 'p'):
        x = (a[f'{k}_r'] - b[f'{k}_r']) / a[f'{k}_sig']
        grid = {}
        for sky, xs in x.groupby(a.sky):
            grid[sky] = dict(m=round(xs.mean(), 2), e=round(xs.std(ddof=1) / np.sqrt(len(xs)), 2) if len(xs) > 1 else None, n=len(xs))
        pair[k] = dict(m=round(x.mean(), 2), e=round(x.std(ddof=1) / np.sqrt(len(x)), 2), n=len(x), grid=grid)
    out['paired'] = pair
    return out


if __name__ == '__main__':
    print(json.dumps(summary(sys.argv[1], os.path.dirname(os.path.abspath(sys.argv[1]))), indent=1))
