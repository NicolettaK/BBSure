"""
Tables of a validation run for the results page, from the output of
scripts/collect.py (results_unblinded.csv, profile_results.csv) and the logs
of every set.

  python plots/validation_page/tables.py <root>

<root> holds runs/set_*/tree/..., results_unblinded.csv and
profile_results.csv. Writes in <root>:
  final_table.csv     one row per set: tree and profile scheme side by side,
                      undecided choices, M_1,0, chain of Res_r0
  tree_full.csv       results_unblinded.csv with the 68% interval of the tree
                      also for the upper limits
  profile_as_tree.csv profile_results.csv with the columns of the tree
                      (r_med, r_std, r_lo1, r_hi1, r_95), for plot_r_sets.py
  first_fit.csv       r of the first fit of the tree (M_none_r, no moments)
                      of every set, the reference of plot_r_sets.py --ref
and fills pte_rstar in profile_results.csv (PTE at r = r* of the final
model, from each result.json).
"""
import json
import os
import sys

import numpy as np
import pandas as pd


def tree_interval(root, s, m):
    """68% interval of r of the tree's final model, around its median m: from
    the fit node of the decision log with that median, otherwise the widths of
    the posterior stored in the profile result (another PolyChord run of the
    same model)."""
    ev = json.load(open(os.path.join(root, 'runs', s, 'tree',
                                     'decision_log.json')))
    for e in ev:
        if e['node'] == 'fit' and e.get('r_lo1') is not None \
                and e.get('r_med') is not None and abs(e['r_med'] - m) < 1e-12:
            return e['r_lo1'], e['r_hi1']
    p = json.load(open(os.path.join(root, 'runs', s, 'tree', 'profile',
                                    'result.json')))['bayes']['posterior'][
        'r_tensor']
    return m - (p['med'] - p['lo1']), m + (p['hi1'] - p['med'])


def setname(i, kind):
    return f"set_{int(i):03d}_{kind}"


def main(R):
    # PTE at r* in the profile table (dof as integers: a set not done would
    # turn the column into floats)
    p = pd.read_csv(f'{R}/profile_results.csv')
    x = []
    for _, r in p.iterrows():
        f = f"{R}/runs/{setname(r['index'], r.kind)}/tree/profile/result.json"
        x.append(json.load(open(f)).get('pte_at_rstar', np.nan)
                 if r.status == 'done' and os.path.isfile(f) else np.nan)
    p['pte_rstar'] = x
    if 'dof_rstar' in p:
        p['dof_rstar'] = p['dof_rstar'].astype('Int64')
    p.to_csv(f'{R}/profile_results.csv', index=False)

    t = pd.read_csv(f'{R}/results_unblinded.csv')

    # tree with the 68% interval also for the upper limits
    tf = t.copy()
    for i, a in tf.iterrows():
        if a.status == 'done' and not np.isfinite(a.r_lo1):
            tf.loc[i, 'r_lo1'], tf.loc[i, 'r_hi1'] = tree_interval(
                R, setname(a['index'], a.kind), a.r_med)
    tf.to_csv(f'{R}/tree_full.csv', index=False)

    # first fit of the tree (M_none_r, no moments), done in every set
    ff = []
    for _, a in t.iterrows():
        f = f"{R}/runs/{setname(a['index'], a.kind)}/tree/decision_log.json"
        e = json.load(open(f))[0] if a.status == 'done' else {}
        ff.append(dict(index=a['index'], kind=a.kind, status=a.status,
                       model=e.get('model', ''), r_med=e.get('r_med', np.nan),
                       r_std=e.get('r_std', np.nan)))
    pd.DataFrame(ff).to_csv(f'{R}/first_fit.csv', index=False)

    # profile scheme with the column names of the tree
    pa = p.merge(t[['index', 'kind', 'sky', 'r_true']], on=['index', 'kind'])
    pa = pa.rename(columns={'r_med_pl': 'r_med', 'r_std_pl': 'r_std',
                            'r_lo1_pl': 'r_lo1', 'r_hi1_pl': 'r_hi1',
                            'r_95_pl0': 'r_95'})
    pa.to_csv(f'{R}/profile_as_tree.csv', index=False)

    # one row per set, tree and profile scheme
    rows = []
    for _, a in t.iterrows():
        b = p[(p['index'] == a['index']) & (p.kind == a.kind)].iloc[0]
        if a.status != 'done' or b.status != 'done':
            continue
        s = setname(a['index'], a.kind)
        tdir = f'{R}/runs/{s}/tree'
        ct = json.load(open(f'{tdir}/conclusion_tree.json'))
        ev = json.load(open(f'{tdir}/decision_log.json'))
        m10 = [e['answer'] for e in ev if e['node'] == 'm10']
        ch = [e for e in ev if e['node'] == 'r0chain']
        und = ct.get('undecided_choices', [])
        kinds = [c['kind'] for c in und]
        res = json.load(open(f'{tdir}/profile/result.json'))
        rt = a.r_true
        rows.append(dict(
            set=s, index=int(a['index']), kind=a.kind, sky=a.sky, r_true=rt,
            t_concl=a.conclusion, t_model=a.model, t_r=a.r_med, t_sig=a.r_std,
            t_lo=a.r_lo1, t_hi=a.r_hi1, t_r95=a.r_95, t_pull=a.pull,
            t_pte=a.pte, t_lowpte=bool(a.low_pte),
            u_mom=sum(k.startswith('M_1,r') for k in kinds),
            u_r0=sum(k.startswith('M_1,0') for k in kinds),
            u_prune=sum(k == 'pruning' for k in kinds),
            u_txt='; '.join(f"{c['kind']}: {c['chosen']} over {c['other']} "
                            f"({c['lnB']:.2f}±{c['lnB_err']:.2f})"
                            for c in und),
            m10=m10[0] if m10 else '',
            chain='+'.join(ch[-1]['families']) if ch else '',
            p_concl=b.conclusion, p_model=b.model_final, p_r=b.r_med_pl,
            p_sig=b.r_std_pl, p_lo=b.r_lo1_pl, p_hi=b.r_hi1_pl,
            p_r95=b.r_95_pl0, p_pull=(b.r_med_pl - rt) / b.r_std_pl,
            p_pte=b.pte_rstar, p_lowpte=bool(b.low_pte),
            p_und=bool(b.undecided),
            p_und_txt='; '.join(f"{c['chosen']} over {c['other']} "
                                f"({c['lnB']:.2f}±{c['lnB_err']:.2f})"
                                for c in res.get('undecided_choices', []))))
    d = pd.DataFrame(rows)
    d['flag_any'] = (d.t_lowpte | (d.u_mom > 0) | (d.u_r0 > 0)
                     | (d.u_prune > 0) | d.p_lowpte | d.p_und)
    d.to_csv(f'{R}/final_table.csv', index=False)
    print(f"{len(d)} sets -> {R}/final_table.csv, tree_full.csv, "
          "profile_as_tree.csv, first_fit.csv")


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]))
