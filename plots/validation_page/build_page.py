"""
Results page of a validation run (HTML, one file plus the images next to it).

  python plots/validation_page/build_page.py <root> <run text> <out.html>
         [--standalone]

<root>: the run, after tables.py (final_table.csv, profile_results.csv,
results_unblinded.csv, runs/set_*/tree/...). <run text>: the text of the run
(title, byline and setup, main results), as run_text_example.html.
The page expects flowchart.png, figures/, paths/ and corners/ next to it
(make_page.sh puts them there). --standalone: a complete HTML document (for a
web server, e.g. the NERSC portal) instead of the fragment of a claude.ai
artifact.
"""
import json, os, re, sys, math, time
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbsure.models import MOMENT_CROSS, MOMENT_GROUPS, model_from_label

TERM = {'00': 'A', '10': r'\beta', '20': r'\beta^2', '01': 'T'}

def spec_tex(pref, spec):
    a, b = spec[1:3], spec[4:6]
    ca, cb = ('d', 's') if pref == 'ds' else (pref, pref)
    return f"{TERM.get(a, a)}_{ca}\\times {TERM.get(b, b)}_{cb}"

def param_tex(n):
    """BBCompSep parameter name -> TeX."""
    fixed = {'r_tensor': 'r', 'beta_d': r'\beta_d', 'beta_s': r'\beta_s', 'temp_d': 'T_d',
             'epsilon_ds': r'\varepsilon_{ds}', 'A_lens': r'A_{\rm lens}'}
    if n in fixed:
        return fixed[n]
    m = n.split('_')
    if len(m) == 3 and m[2] == 'bb' and m[0] in ('amp', 'alpha'):
        return ('A' if m[0] == 'amp' else r'\alpha') + f'_{m[1]}^{{BB}}'
    if len(m) == 3 and m[0] in ('A', 'alpha') and m[2].startswith('w'):
        return ('A' if m[0] == 'A' else r'\alpha') + r'\,[' + spec_tex(m[1], m[2]) + ']'
    return n.replace('_', r'\_')

def model_tex(lab):
    """Moment spectra of the model, in TeX ('' = no moments)."""
    if not isinstance(lab, str) or not lab.startswith('M_'):
        return ''
    M = model_from_label(lab)
    out = []
    for g in sorted(M.moments):
        G = MOMENT_GROUPS[g]
        out += [spec_tex(G['pref'], x) for x in G['spectra']]
    for pair, X in MOMENT_CROSS.items():
        if set(pair) <= M.moments:
            out += [spec_tex(X['pref'], x) for x in X['spectra']]
    return ',\\; '.join(out)

def model_specs(lab):
    """Moment spectra of a model, as a list of TeX strings."""
    t = model_tex(lab)
    return t.split(',\\; ') if t else []

def added_tex(base, lab):
    """Spectra of lab that are not in base, in TeX."""
    b = set(model_specs(base))
    return ',\\; '.join(x for x in model_specs(lab) if x not in b)

def r0_rounds(ev):
    """Rounds of the Res_r0 chain: model with r = 0 and its ln B against M_0,r."""
    out = []
    for i, e in enumerate(ev):
        q = e.get('question', '')
        if e['node'] != 'pref' or not q.startswith('Is M_1,0 ('):
            continue
        m0 = q.split('(')[1].split(')')[0]
        base = q.split('(')[2].split(')')[0]
        m1r = m0[:-2] + '_r'
        r1 = [x for x in ev[:i] if x['node'] == 'pref' and x.get('question', '').startswith(f'Is M_1,r ({m1r})')]
        lr = r1[-1] if r1 else None
        out.append(dict(add=added_tex(base, m0), all=model_tex(m0), base=model_tex(base),
                        br=lr and round(lr['lnB'], 2), er=lr and round(lr['lnB_err'], 2), ar=lr and lr['answer'],
                        b0=round(e['lnB'], 2), e0=round(e['lnB_err'], 2), a0=e['answer']))
    return out
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
TABLE = os.path.join(os.path.abspath(ARGS[0]), 'final_table.csv')
d = pd.read_csv(TABLE)
def f(x, n=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)
import os
from tables import tree_interval
ROOT = os.path.dirname(TABLE)
KIND = {'M_1,r vs M_0,r': 'mom', 'M_1,0 vs M_0,r': 'r0', 'pruning': 'cut'}

RANK = {'': 0, '✗✗': 0.5, '✗': 1, '✓': 2, '✓✓': 3}
# r significantly below zero (a moment subtracting too much): r / sigma < NEG
NEG = -1.5


def r0grade(x, e, xx=False):
    """x = ln B(model with r = 0 vs M_0,r): preferred by a lot (✓✓), by a
    little (✓, |ln B| < max(1, 2 sigma): inconclusive on the Jeffreys scale
    of Trotta 2008, or within the numerical error), not preferred by a
    little (✗), not preferred by a lot (✗✗, with xx). Every set gets a
    grade from its final model with r = 0 (M_0,0)."""
    if x is None or e is None:
        return ''
    b = max(1., 2 * e if e == e else 0.)
    if x > 0:
        return '✓✓' if x >= b else '✓'
    return '✗' if x > -b else ('✗✗' if xx else '')


def details(name):
    """Res_r0 outcome and undecided choices of a set, from its logs."""
    t = os.path.join(ROOT, 'runs', name, 'tree')
    ev = json.load(open(os.path.join(t, 'decision_log.json')))
    ct = json.load(open(os.path.join(t, 'conclusion_tree.json')))
    res = json.load(open(os.path.join(t, 'profile', 'result.json')))
    reached = any(e['node'] == 'resid' and e['question'].startswith('Res_r0') for e in ev)
    ch = [e for e in ev if e['node'] == 'r0chain']
    early = any(e['node'] == 'r1sigma' and e['answer'] == 'Yes' for e in ev)
    # r (median, sigma; 1e-3) of each model: posterior of its evidence run in
    # the tree, profile of the model in the profile scheme
    rt_ = {e['model']: (e['r_med'], e['r_std']) for e in ev
           if e['node'] == 'fit' and e.get('model') and e.get('r_std') is not None
           and isinstance(e.get('logZ'), float) and e['logZ'] == e['logZ']}
    pf = os.path.join(t, 'profile', 'decision_log.json')
    rp_ = {e['question'].split()[0]: (e['r_med_pl'], e['r_std_pl'])
           for e in (json.load(open(pf)) if os.path.isfile(pf) else [])
           if e['node'] == 'compare' and 'r_med_pl' in e}

    def rr(lab, tab):
        if lab.endswith('_0'):
            return '0'
        v = tab.get(lab)
        return f"{1e3 * v[0]:.2f} ± {1e3 * v[1]:.2f}" if v else None
    und = [dict(k=KIND.get(c['kind'], c['kind']), c=model_tex(c['chosen']),
                o=model_tex(c['other']), c0=c['chosen'].endswith('_0'),
                o0=c['other'].endswith('_0'), b=round(c['lnB'], 2), e=round(c['lnB_err'], 2),
                rc=rr(c['chosen'], rt_), ro=rr(c['other'], rt_))
           for c in ct.get('undecided_choices', [])]
    pund = [dict(c=model_tex(c['chosen']), o=model_tex(c['other']),
                 b=round(c['lnB'], 2), e=round(c['lnB_err'], 2),
                 rc=rr(c['chosen'], rp_), ro=rr(c['other'], rp_)) for c in res.get('undecided_choices', [])]
    und, pund = dedup(und, pund)
    p0 = phase0_info(t, ev)
    b = res['bayes']
    names = ['r_tensor'] + [n for n in b['posterior'] if n != 'r_tensor']
    names += [n for n in res['params_at_rstar'] if n not in names]
    pars = []
    for n in names:
        po, pp = b['posterior'].get(n), res['params_at_rstar'].get(n)
        pars.append(dict(t=param_tex(n),
                         tb=b['best_fit'].get(n), tm=po and po['med'], ts=po and po['std'],
                         tl=po and po.get('lo1'), th=po and po.get('hi1'),
                         pb=res['best_fit_at_rstar'].get(n),
                         pm=(res['r_med_pl'] if n == 'r_tensor' else pp and pp['med']),
                         ps=(res['r_std_pl'] if n == 'r_tensor' else pp and pp['std']),
                         pl=(res['r_lo1_pl'] if n == 'r_tensor' else pp and pp.get('lo1')),
                         ph=(res['r_hi1_pl'] if n == 'r_tensor' else pp and pp.get('hi1'))))
    same = b['model'] == res['model_final']
    rb = b['posterior'].get('r_tensor', {})
    return dict(pars=pars, same=same, blo=rb.get('lo1'), bhi=rb.get('hi1'), tbr=b['best_fit'].get('r_tensor'), pbr=res['r_star_final'],
                r0=('early' if early else 'none' if not reached else 'nothing' if not ch else 'chain'),
                fam=ch[-1]['families'] if ch else [], und=und, pund=pund,
                cands=res.get('candidates', []), start=res.get('start', ''), rounds=r0_rounds(ev),
                lb0=ct.get('lnB_rfree_vs_r0'), lb0e=ct.get('lnB_err'), r0c=r0comps(ev, ct), p0=p0)


def dedup(und, pund):
    """The same borderline choice (same two models) met more than once, e.g.
    in Res_wr and again in Res_r0, or in the tree and again in the profile
    scheme: shown once, with n = how many times and the range of ln B."""
    seen = {}
    for c in und:
        key = (c['c'], c['o'], c['c0'], c['o0'])
        if key in seen:
            s = seen[key]
            s['n'] += 1
            s['bmin'], s['bmax'] = min(s['bmin'], c['b']), max(s['bmax'], c['b'])
        else:
            seen[key] = dict(c, n=1, bmin=c['b'], bmax=c['b'], prof=False)
    out = list(seen.values())
    pout = []
    for c in pund:
        same = [s for s in out if (s['c'], s['o']) in ((c['c'], c['o']), (c['o'], c['c']))
                and not s['c0'] and not s['o0']]
        if same:
            same[0]['prof'] = True
        else:
            pout.append(c)
    return out, pout


def phase0_info(t, ev):
    """Phase 0 check (run_tree.py --phase0): its model, the spectra it keeps
    that are not in the final model of the analysis, and the decisions of
    the two halves. None if the set had no phase 0."""
    f = os.path.join(t, 'conclusion.json')
    p = json.load(open(f)).get('phase0') if os.path.isfile(f) else None
    if not p:
        return None
    steps = [dict(h=e.get('split'), q=model_tex(e['question'].split('(')[1].split(')')[0]),
                  a=e['answer'], b=round(e['lnB'], 2), ml=e.get('method') == 'ML estimate')
             for e in ev if e['node'] == 'phase0_pref']
    kept = {e.get('split'): model_tex(e['answer']) for e in ev if e['node'] == 'phase0_split'}
    extra = [model_tex('M_' + g + '_r') for g in p.get('extra_groups', [])]
    return dict(start=model_tex(p['start']), extra=extra, steps=steps, kept=kept)


def r0comps(ev, ct):
    """Models with r = 0 compared with M_0,r: the M_1,0 of Res_r0 (PolyChord
    comparisons only) and the final model with r = 0 (M_0,0)."""
    out = []
    for e in ev:
        q = e.get('question', '')
        if e['node'] == 'pref' and q.startswith('Is M_1,0 (') and 'lnB' in e and not e.get('method'):
            m = q.split('(')[1].split(')')[0]
            out.append(dict(w='M₁,₀', m=model_tex(m), x=round(e['lnB'], 2), e=round(e['lnB_err'], 2),
                            g=r0grade(e['lnB'], e['lnB_err'], xx=True)))
    lb, le = ct.get('lnB_rfree_vs_r0'), ct.get('lnB_err')
    if lb is not None:
        out.append(dict(w='M₀,₀', m=model_tex(ct.get('model')), x=round(-lb, 2), e=round(le, 2),
                        g=r0grade(-lb, le, xx=True)))
    return out

fin = lambda v: isinstance(v, float) and math.isfinite(v)
rows = []
for _, r in d.iterrows():
    x = details(r.set)
    if not fin(r.t_lo):
        r.t_lo, r.t_hi = tree_interval(ROOT, r.set, r.t_r)
    # r of the tree in the parameter table: the same numbers as in the main table
    x['pars'][0].update(tm=r.t_r, ts=r.t_sig, tl=r.t_lo, th=r.t_hi)
    rows.append(dict(
        s=r.set, i=int(r['index']), k=r.kind, sky=r.sky, rt=f(r.r_true * 1e3, 2),
        tc=r.t_concl, tm=r.t_model, tmx=model_tex(r.t_model), pmx=model_tex(r.p_model),
        m10x=model_tex(r.m10) if isinstance(r.m10, str) else '', tr=f(r.t_r * 1e3, 2), ts=f(r.t_sig * 1e3, 2), tlo=f(r.t_lo * 1e3, 2), thi=f(r.t_hi * 1e3, 2),
        t95=f(r.t_r95 * 1e3, 2) if isinstance(r.t_r95, float) and math.isfinite(r.t_r95) else None,
        tp=f(r.t_pull, 2), tpte=f(r.t_pte, 3), tlow=bool(r.t_lowpte),
        um=int(r.u_mom), ur=int(r.u_r0), up=int(r.u_prune),
        ut='' if pd.isna(r.u_txt) else r.u_txt, m10='' if pd.isna(r.m10) else r.m10,
        ch='' if pd.isna(r.chain) else r.chain,
        pc=r.p_concl, pm=r.p_model, pr=f(r.p_r * 1e3, 2), ps=f(r.p_sig * 1e3, 2), plo=f(r.p_lo * 1e3, 2), phi=f(r.p_hi * 1e3, 2),
        p95=f(r.p_r95 * 1e3, 2), pp=f(r.p_pull, 2), ppte=f(r.p_pte, 3),
        plow=bool(r.p_lowpte), pu=bool(r.p_und), put='' if pd.isna(r.p_und_txt) else r.p_und_txt,
        tw=bool(r.r_true > r.t_r95 if r.t_concl == 'upper_limit' else r.r_true == 0),
        pw=bool(r.r_true > r.p_r95 if r.p_concl == 'upper_limit' else r.r_true == 0),
        r0g=max((c['g'] for c in x['r0c']), key=lambda g: RANK[g], default=''),
        p0=bool(x['p0'] and x['p0']['extra']),
        tneg=bool(r.t_r / r.t_sig < NEG), pneg=bool(r.p_r / r.p_sig < NEG),
        # distance of r = 0 in sigma: posterior median / standard deviation
        # (tree), the same from the profile posterior (profile scheme)
        tz=f(r.t_r / r.t_sig, 1), pz=f(r.p_r / r.p_sig, 1),
        x=x))
from summary import summary


def block(text, name):
    """Block <!--NAME--> of the run text (up to the next block)."""
    i = text.index(f'<!--{name}-->') + len(f'<!--{name}-->')
    j = text.find('<!--', i)
    return text[i:j if j >= 0 else None].strip('\n') + '\n'


run = open(ARGS[1]).read()
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'template.html')).read()
# blind run (no true r in the table): sections that need the true r or the
# sky (<!--TRUTH--> ... <!--/TRUTH-->) out, the blind ones (<!--BLIND-->) in;
# otherwise the reverse
BLIND = bool(d.r_true.isna().all())
html = re.sub(r'<!--%s-->.*?<!--/%s-->\n?' % (('TRUTH',) * 2 if BLIND else ('BLIND',) * 2), '', html, flags=re.S)
html = re.sub(r'<!--/?(TRUTH|BLIND)-->\n?', '', html)
html = html.replace('/*BLIND*/false', 'true' if BLIND else 'false')
# review of the sets (review.csv in the root, exported from the page: set,
# conclusion_trusted, model_trusted, comment): fixed in the page, read only,
# with the plots of make_page.sh instead of the ones drawn in the browser
RF = os.path.join(ROOT, 'review.csv')
REV = None
if os.path.isfile(RF):
    rv = pd.read_csv(RF, dtype=str, keep_default_na=False)
    REV = dict(ok=[r.set for r in rv.itertuples() if r.conclusion_trusted == '1'],
               okm=[r.set for r in rv.itertuples() if r.model_trusted == '1'],
               notes={r.set: r.comment for r in rv.itertuples() if r.comment.strip()})
html = re.sub(r'<!--%s-->.*?<!--/%s-->\n?' % (('LIVE',) * 2 if REV else ('FROZEN',) * 2), '', html, flags=re.S)
html = re.sub(r'<!--/?(LIVE|FROZEN)-->\n?', '', html)
html = html.replace('/*REVIEW*/null', json.dumps(REV, ensure_ascii=False).replace('</', '<\\/') if REV else 'null')
html = (html.replace('<!--TITLE-->', block(run, 'TITLE').strip())
        .replace('<!--HEADER-->\n', block(run, 'HEADER'))
        .replace('<!--KEYS-->\n', block(run, 'KEYS'))
        .replace('/*DATA*/[]', json.dumps(rows, separators=(',', ':')))
        .replace('/*SUMMARY*/{}', json.dumps(summary(TABLE, ROOT),
                                             separators=(',', ':'))))
# build tag on every image URL (?v=...): a rebuilt page does not show the
# images of the previous build kept in the browser cache
build = time.strftime('%Y%m%d%H%M%S')
html = re.sub(r'src="((?:figures/[^"?]+|flowchart)\.png)"',
              rf'src="\1?v={build}"', html)
html = html.replace('<script>', f"<script>const BUILD = '{build}';", 1) \
    if '<script>' in html else html
if '--standalone' in sys.argv:
    html = html.replace('<meta name="claude:viewport-fit-sides" '
                        'content="cover">\n', '')
    i = html.index('</style>') + len('</style>')
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, '
            'initial-scale=1">\n' + html[:i] + '\n</head>\n<body>\n'
            + html[i:] + '\n</body>\n</html>\n')
# figures not made for this run (e.g. the d9s4 ones in a run with random
# skies only): out of the page
OUTD = os.path.dirname(os.path.abspath(ARGS[2]))
html = re.sub(r'\s*<figure><img data-lb src="(figures/[^"?]+)[^"]*"[^>]*>\s*<figcaption>.*?</figcaption></figure>',
              lambda m: m.group(0) if os.path.isfile(os.path.join(OUTD, m.group(1)))
              and not (m.group(1).endswith('_d9s4.png') and (d.kind != 'd9s4').all()) else '', html, flags=re.S)
open(ARGS[2], 'w').write(html)
print(len(rows), 'sets ->', ARGS[2])
