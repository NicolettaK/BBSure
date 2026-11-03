"""
Path of one tree through the flowchart: one box per node of
decision_log.json (question, answer, key numbers), the tree on the left and
the profile scheme on the right (if it was run).

  python plots/plot_tree_path.py <runs_dir of the tree> [--profile DIR]
         [--out tree_path.pdf] [--title "..."]

The profile scheme is read from <runs_dir>/profile (the default of
scripts/run_tree.py) or from --profile. Writes <out> (pdf) and the same
figure as png.
Colours as in docs/flowcharts/decision_tree: green = new model, purple =
action (residual test, pruning, comparison), yellow = question on chi2 or r,
red = model comparison, blue = upper limit, orange = r detection.
"""
import argparse
import json
import os
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy import stats  # noqa: E402

from bbsure.models import (MOMENT_CROSS, MOMENT_GROUPS,  # noqa: E402
                           amp_name, model_from_label)

COLORS = {'green': ('#2f9e44', '#ebfbee'), 'purple': ('#9c36b5', '#f8f0fc'),
          'yellow': ('#e0a800', '#fff9db'), 'red': ('#e03131', '#fff5f5'),
          'blue': ('#1971c2', '#e7f5ff'), 'orange': ('#e8590c', '#fff4e6'),
          'grey': ('#868e96', '#f1f3f5')}
LABEL = re.compile(r'M_[A-Za-z0-9+\-_]+?_(?:r|0)\b')
WRAP = 140          # characters per line (of the mathtext source)
LH = 1.0            # line height (data units)
PAD = 0.6           # vertical padding inside a box
GAP = 1.4           # vertical gap between boxes (arrow)
WIDTH = 36.         # box width (data units)


# order (beta, T) of a moment map -> name, for the component of the prefix
TERM = {'00': 'A', '10': r'\beta', '20': r'\beta^2', '01': 'T'}

# plain text of the logs -> mathtext (applied outside $...$ only)
TEX = [(r'M_(\d),(r|0)', r'$M_{\1,\2}$'), (r'M\*', r'$M^*$'),
       (r'\bchi2\b', r'$\\chi^2$'), (r'\bln ([ZB])\b', r'$\\ln \1$'),
       (r'\bRes_wr\b', r'Res$_{\\rm wr}$'),
       (r'\bRes_r0\b', r'Res$_{r=0}$'), (r'\br_min\b', r'$r_{\\rm min}$'),
       (r'\bsigma\b', r'$\\sigma$'), (r'(?<![\w$])r(?![\w$])', r'$r$'),
       (r'<- ', '← '), (r'-> ', '→ ')]


def tex(s):
    """Apply TEX to the parts of s outside $...$."""
    for pat, rep in TEX:
        seg = s.split('$')
        s = '$'.join(re.sub(pat, rep, x) if i % 2 == 0 else x
                     for i, x in enumerate(seg))
    return s


def spec_name(pref, spec):
    """w10w10 of d -> beta_d x beta_d; w00w10 of ds -> A_d x beta_s (dust x
    synch cross terms: the first map is the dust one, the second the synch
    one), in mathtext."""
    a, b = spec[1:3], spec[4:6]
    ca, cb = ('d', 's') if pref == 'ds' else (pref, pref)
    return f"${TERM.get(a, a)}_{ca}\\times {TERM.get(b, b)}_{cb}$"


def short(lab):
    """M_d+s-d_w00w10_r -> 'd [beta_d x beta_d] + s' (mathtext): the moment
    families with the spectra left after the pruning (a family without
    brackets is complete), ', r = 0' if r is fixed."""
    if not lab or not LABEL.fullmatch(lab):
        return str(lab)
    M = model_from_label(lab)

    def names(pref, specs):
        return ', '.join(spec_name(pref, x) for x in specs)

    parts = []
    for g in sorted(M.moments):
        G = MOMENT_GROUPS[g]
        kept = [x for x in G['spectra']
                if amp_name(G['pref'], x) not in M.off]
        parts.append(g if len(kept) == len(G['spectra'])
                     else f"{g} [{names(G['pref'], kept) or 'none'}]")
    for pair, X in MOMENT_CROSS.items():
        # cross spectra between two groups: shown only if pruned
        if set(pair) <= M.moments:
            gone = [x for x in X['spectra']
                    if amp_name(X['pref'], x) in M.off]
            if gone:
                parts.append(f"(no {names(X['pref'], gone)})")
    s = ' + '.join(parts) or 'no moments'
    return s + (', $r = 0$' if not M.r_free else '')


def shorten(text):
    return LABEL.sub(lambda m: short(m.group(0)), text)


def pm(e, k, err, f='.4f'):
    return f"{e[k]:{f}} ± {e[err]:{f}}"


def tree_items(ev):
    """(colour, [lines], answer) for the nodes of the tree."""
    items, pending = [], []
    for e in ev:
        n, q, a = e['node'], e['question'], e['answer']
        if n == 'fit':
            # a new M_0,r: the first fit of each iteration
            if q == f"fit of {e.get('model')}":
                nums = (f"r = {pm(e, 'r_med', 'r_std')},  "
                        f"ln Z = {e['logZ']:.1f}")
                if items and items[-1][1][0].startswith('M_0,r = M_1,r'):
                    # the new M_0,r just adopted: its numbers in that box
                    items[-1][1][1:] = [nums]
                else:
                    items.append(('green', [f"M_0,r = {short(e['model'])}",
                                            nums], None))
        elif n == 'chi2':
            items.append(('yellow', ['is chi2 good?',
                                     f"chi2 = {e['chi2']:.1f} / {e['dof']},"
                                     f"  PTE = {e['pte']:.3g}"], a))
        elif n == 'r1sigma':
            items.append(('yellow', ['is r compatible with zero at 1 sigma?',
                                     f"68%: [{e['r_lo1']:.4f}, "
                                     f"{e['r_hi1']:.4f}]"], a))
        elif n == 'resid_fit':
            pending.append(e)
        elif n == 'resid':
            name = q.split(':')[0]
            head = f"{name}: residuals of {short(e['base'])}" \
                if e.get('base') else name
            lines = [head]
            for p in pending:
                g = re.search(r'moment (\S+) on', p['question'])
                g = g.group(1) if g else '?'
                s = f"  {g}: {p['answer']}"
                if p.get('stage') == 'lrt':
                    s += f"  (Δχ² = {p['dchi2']:.1f}, p = {p['pvalue']:.2g})"
                elif p['answer'] == 'detected' and p.get('logZ') is not None:
                    s += f"  (residual ln Z = {p['logZ']:.1f})"
                lines.append(s)
            pending = []
            items.append(('purple', lines + ['moment detected?'], a))
        elif n in ('pref', 'prune'):
            lines = [shorten(q)]
            if 'lnB' in e:
                s = f"ln B = {e['lnB']:.2f} ± {e['lnB_err']:.2f}"
                if max(e.get('pc_runs', [1])) > 1:
                    s += f"  (PolyChord runs {e['pc_runs']})"
                lines.append(s)
            items.append(('red', lines, a))
        elif n == 'prune_check':
            if e.get('undetected'):
                items.append(('purple', ['pruning: amplitudes with 0 in '
                                         'their 68% HPD',
                                         ', '.join(e['undetected'])], None))
        elif n == 'replace':
            src = q[q.find('(') + 1:q.rfind(')')] if '(' in q else ''
            items.append(('green', [f"M_0,r = M_1,r = {short(a)}"
                                    + (f"  (from {src[5:]})" if src else ''),
                                    ''], None))
        elif n == 'cycle':
            if a == 'Yes':
                items.append(('grey', [shorten(q)], a))
        elif n == 'r3sigma':
            items.append(('yellow', ['is r compatible with zero at 3 sigma?',
                                     f"99.7%: [{e['r_lo3']:.4f}, "
                                     f"{e['r_hi3']:.4f}]"], a))
        elif n == 'END':
            fit = [f for f in ev if f['node'] == 'fit'
                   and f.get('model') == e.get('model') and 'chi2' in f]
            chi = (fit[-1]['chi2'], fit[-1]['dof']) if fit else None
            items.append(end_item(e, short(e.get('model')), chi))
        else:
            items.append(('grey', [shorten(q)], a))
    return items


def end_item(e, model, chi=None, where=''):
    """Conclusion box; chi = (chi2, dof) of the final model."""
    extra = [] if chi is None else \
        [f"chi2{where} = {chi[0]:.1f} / {chi[1]},  "
         f"PTE = {stats.chi2.sf(chi[0], chi[1]):.3g}"]
    if e.get('conclusion') == 'upper_limit':
        r95 = e.get('r_95', e.get('r_95_pl0'))
        return ('blue', ['UPPER LIMIT', model, f"r < {r95:.4f} (95%)"]
                + extra, None)
    if e.get('conclusion') == 'r_detection':
        med = e.get('r_med', e.get('r_med_pl'))
        lo = e.get('r_lo1', e.get('r_lo1_pl'))
        hi = e.get('r_hi1', e.get('r_hi1_pl'))
        return ('orange', ['r DETECTION', model,
                           f"r = {med:.4f} [{lo:.4f}, {hi:.4f}] (68%)"]
                + extra, None)
    return ('grey', [shorten(e['question'])], e['answer'])


def profile_items(ev, res, ndata=None):
    """(colour, [lines], answer) for the nodes of the profile scheme;
    ndata: size of the data vector (for the dof at r = r_min)."""
    comp = (res or {}).get('comparison') or {}
    items = []
    if res:
        cands = ', '.join(res.get('candidates') or []) or 'none'
        items.append(('green', [f"M* = {short(res['model_tree'])}",
                                f"candidates from the last Res_r0: {cands}"],
                      None))
    rows = [e for e in ev if e['node'] == 'compare']
    if rows:
        best = max((e for e in rows if 'logZ' in e),
                   key=lambda e: comp.get(e['question'].split()[0], e)
                   ['logZ'], default=None)
        lines = ['compare, each model at its own r (local evidence)']
        for e in rows:
            lab = e['question'].split()[0]
            if 'logZ' not in e:
                lines.append(f"  {short(lab)}: {e['answer']}")
                continue
            c = comp.get(lab, e)
            s = f"  {short(lab)}: r = {e['r_min']:.4f}, " \
                f"ln Z = {c['logZ']:.2f} ± {c['logZerr']:.2f}"
            if len(c.get('logZ_runs') or []) > 1:
                s += f" ({len(c['logZ_runs'])} runs)"
            lines.append(s + ('  <- best' if e is best else ''))
        items.append(('purple', lines, None))
    for e in ev:
        n, q, a = e['node'], e['question'], e['answer']
        if n == 'final':
            items.append(('red', ['final model: ' + short(a)], None))
        elif n == 'post':
            items.append(('purple', ['other parameters: emcee with r fixed '
                                     'at the minimum of the profile'], None))
        elif n == 'r3sigma':
            items.append(('yellow', ['is r compatible with zero at 3 sigma?',
                                     '(posterior from the profile)',
                                     f"99.7%: [{e['r_lo3']:.4f}, "
                                     f"{e['r_hi3']:.4f}]"], a))
        elif n == 'END':
            # chi2 of the final model at the minimum of its profile; free
            # parameters = those of the posterior at fixed r ('post' node)
            post = [p for p in ev if p['node'] == 'post']
            npar = len([k for k in post[-1] if k.endswith('_med')]) \
                if post else None
            chi = (e['chi2_at_rstar'], ndata - npar) \
                if ndata and npar and 'chi2_at_rstar' in e else None
            items.append(end_item(e, short(e.get('model_final')), chi,
                                  ' at r_min'))
        elif n != 'compare':
            items.append(('grey', [shorten(q)], a))
    return items


def wrap(lines):
    """Lines in mathtext, wrapped (never inside $...$); the first bold."""
    out = []
    for i, s in enumerate(lines):
        seg = tex(s).split('$')
        s = '$'.join(x.replace(' ', '\0') if k % 2 else x
                     for k, x in enumerate(seg))
        w = textwrap.wrap(s, WRAP, subsequent_indent='    ') or ['']
        out += [(t.replace('\0', ' '), i == 0) for t in w]
    return out


def draw_column(ax, items, x0, title):
    """Boxes from the top (y = 0) downwards; returns the boxes (x, y, h)."""
    ax.text(x0 + WIDTH / 2, 1.2, title, ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    y, boxes = 0., []
    for col, lines, ans in items:
        wl = wrap(lines)
        if ans is not None:
            wl.append((f"→ {ans}", True))
        h = len(wl) * LH + 2 * PAD
        edge, face = COLORS[col]
        ax.add_patch(FancyBboxPatch((x0, y - h), WIDTH, h,
                                    boxstyle='round,pad=0,rounding_size=0.5',
                                    ec=edge, fc=face, lw=1.3))
        for k, (t, bold) in enumerate(wl):
            ax.text(x0 + 0.8, y - PAD - (k + 0.5) * LH, t, va='center',
                    ha='left', fontsize=8,
                    fontweight='bold' if bold else 'normal')
        if boxes:
            ax.annotate('', xy=(x0 + WIDTH / 2, y),
                        xytext=(x0 + WIDTH / 2, y + GAP),
                        arrowprops=dict(arrowstyle='->', lw=1.1))
        boxes.append((x0, y, h))
        y -= h + GAP
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tree_dir', help='runs_dir of the tree')
    ap.add_argument('--profile', help='output directory of the profile '
                    'scheme (default: <tree_dir>/profile, if it exists)')
    ap.add_argument('--out', help='output pdf (default: '
                    '<tree_dir>/tree_path.pdf)')
    ap.add_argument('--title', default=None)
    a = ap.parse_args()

    ev = json.load(open(os.path.join(a.tree_dir, 'decision_log.json')))
    columns = [(tree_items(ev), 'tree (Bayesian)')]
    pdir = a.profile or os.path.join(a.tree_dir, 'profile')
    if os.path.isfile(os.path.join(pdir, 'decision_log.json')):
        pev = json.load(open(os.path.join(pdir, 'decision_log.json')))
        rf = os.path.join(pdir, 'result.json')
        res = json.load(open(rf)) if os.path.isfile(rf) else None
        fits = [f for f in ev if f['node'] == 'fit' and 'nfree' in f]
        ndata = fits[0]['dof'] + fits[0]['nfree'] if fits else None
        columns.append((profile_items(pev, res, ndata), 'profile scheme'))

    def height(items):
        return sum(len(wrap(ln)) + (ans is not None) for _, ln, ans in items) \
            * LH + len(items) * (2 * PAD + GAP)

    H = max(height(it) for it, _ in columns) + 4
    W = len(columns) * (WIDTH + 6)
    fig = plt.figure(figsize=(W * 0.24, H * 0.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-2, W - 2)
    ax.set_ylim(-H + 3, 4)
    ax.axis('off')

    cols = [draw_column(ax, it, i * (WIDTH + 6), t)
            for i, (it, t) in enumerate(columns)]
    if len(cols) == 2:
        # from the conclusion of the tree to the profile scheme
        (x, y, h), (x2, y2, h2) = cols[0][-1], cols[1][0]
        xm, ya, yb = (x + WIDTH + x2) / 2, y - h / 2, y2 - h2 / 2
        ax.plot([x + WIDTH, xm, xm], [ya, ya, yb], color='#0b7285', lw=1.1)
        ax.annotate('', xy=(x2, yb), xytext=(xm, yb),
                    arrowprops=dict(arrowstyle='->', lw=1.1,
                                    color='#0b7285'))
    ax.text(-1, 3.4, a.title or os.path.abspath(a.tree_dir), fontsize=9,
            color='#495057', va='top')

    out = a.out or os.path.join(a.tree_dir, 'tree_path.pdf')
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + '.png', dpi=130)
    print('written', out)


if __name__ == '__main__':
    main()
