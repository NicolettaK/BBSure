"""
Path of one tree through the flowchart: one box per node of
decision_log.json (question, answer, key numbers), the tree on the left and
the profile scheme on the right (if it was run).

  python plots/plot_tree_path.py <runs_dir of the tree> [--profile DIR]
         [--out tree_path.pdf] [--title "..."]

The profile scheme is read from <runs_dir>/profile (the default of
scripts/run_tree.py) or from --profile. Writes <out> (pdf) and the same
figure as png.
Colours, notation and step numbers as in the step-by-step flowchart
(docs/flowcharts/step_by_step): blue = fits and actions, cream = questions,
violet = profile scheme, green = upper limit, red = r detection; a step met
more than once gets a letter (10a, 10b, ...).
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
                           RESTORE, amp_name, model_from_label)
from bbsure.profile_scheme import chain_candidates  # noqa: E402

# as the step-by-step flowchart (docs/flowcharts/step_by_step): blue boxes
# = fits and actions of the tree, cream = questions, violet = profile
# scheme, green = upper limit, red = r detection
COLORS = {'box': ('#2a6fdb', '#eaf1fc'), 'dia': ('#d9822b', '#fff4e0'),
          'pbox': ('#7b4fd6', '#f1eafc'), 'ul': ('#2f9e44', '#e6f4ea'),
          'det': ('#d9534f', '#fbe9e7'), 'grey': ('#868e96', '#f1f3f5')}
QUESTION = ('chi2', 'r1sigma', 'resid', 'pref', 'prune', 'cycle', 'r3sigma')
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
       (r'\bRes_r0\b', r'Res$_{\\rm r0}$'), (r'\br_min\b', r'$r_{\\rm min}$'),
       (r'\bM_1\b', r'$M_1$'), (r'\bDelta (AIC|BIC)\b', r'$\\Delta$\1'),
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


def amp_tex(name):
    """A_d_w00w10 -> the moment spectrum A_d x beta_d (mathtext)."""
    _, pref, spec = name.split('_', 2)
    return spec_name(pref, spec)


def fam_tex(g):
    """Moment group (or 'restore-<group>') -> its spectra (mathtext)."""
    if g.startswith(RESTORE):
        return 'restore ' + fam_tex(g[len(RESTORE):])
    G = MOMENT_GROUPS.get(g)
    if G is None:
        return g
    return ', '.join(spec_name(G['pref'], x) for x in G['spectra'])


def fam_left(g, off):
    """Family g without the spectra of off (pruned with r = 0 in Res_r0),
    as tested in the profile scheme: 'beta_d^2 x beta_d^2 (A_d x beta_d^2
    pruned with r = 0)'."""
    G = MOMENT_GROUPS.get(g)
    if G is None or not off:
        return fam_tex(g)
    left = [spec_name(G['pref'], x) for x in G['spectra']
            if amp_name(G['pref'], x) not in off]
    return (', '.join(left) + ' (' + ', '.join(amp_tex(x) for x in off)
            + ' pruned with r = 0)')


def short(lab):
    """M_d+s-d_w00w10_r -> the moment spectra left in the model after the
    pruning, in mathtext ('beta_d x beta_d, A_s x beta_s, ...'), as in the
    results table; ', r = 0' if r is fixed."""
    if not lab or not LABEL.fullmatch(lab):
        return str(lab)
    M = model_from_label(lab)
    parts = []
    for g in sorted(M.moments):
        G = MOMENT_GROUPS[g]
        parts += [spec_name(G['pref'], x) for x in G['spectra']
                  if amp_name(G['pref'], x) not in M.off]
    for pair, X in MOMENT_CROSS.items():
        if set(pair) <= M.moments:
            parts += [spec_name(X['pref'], x) for x in X['spectra']
                      if amp_name(X['pref'], x) not in M.off]
    s = ', '.join(parts) or 'no moments'
    return s + (', $r = 0$' if not M.r_free else '')


def shorten(text):
    return LABEL.sub(lambda m: short(m.group(0)), text)


def pm(e, k, err, f='.4f'):
    return f"{e[k]:{f}} ± {e[err]:{f}}"


def tree_step(e, phase, early):
    """Step of the flowchart of a node of the tree; phase = 'wr' or 'r0'
    (search in progress) or None; early = r compatible with 0 at 1 sigma."""
    n, q = e['node'], e['question']
    wr = phase != 'r0'
    if n == 'fit':
        return '1'
    if n in ('chi2', 'r1sigma'):
        return {'chi2': '2', 'r1sigma': '3'}[n]
    if n == 'resid':
        return '4–5' if q.startswith('Res_wr') else '13–14'
    if n == 'prune_check':
        if q.startswith('Undetected new'):
            return '15' if q.endswith('(r = 0)?') else ('7' if wr else '16')
        if q.endswith('(r = 0)?'):
            return '17'
        return '3' if early else ('9' if wr else '16')
    if n == 'prune':
        ref = q.split('w.r.t.')[-1].strip(' ?')
        return '17' if ref.endswith('_0') else ('9' if wr else '16')
    if n == 'pref':
        if q.startswith('Is M_1,0'):
            return '17'
        return ('8' if e.get('method') else '10') if wr else '16'
    if n in ('replace', 'cycle'):
        return '11'
    if n == 'r0order':
        return '15'
    if n == 'r0chain':
        return '17' if e['answer'] == 'Yes' else '18'
    if n == 'm10':
        return '17'
    if n == 'pruned_out':
        if q.startswith('M_1,0'):
            return '15' if 'with r = 0' in q else '17'
        return ('7' if 'no new moment detected' in q else '9') if wr else '16'
    if n == 'unfittable':
        return '7' if wr else '16'
    if n in ('r3sigma',):
        return '19'
    if n == 'END':
        return '3' if early else '19'
    return ''


def letters(items):
    """Steps met more than once: a, b, c, ... for each visit (consecutive
    boxes of the same step are one visit and share the label)."""
    visit, prev = [], None
    for it in items:
        st = it[4]
        visit.append(st if st != prev else None)
        prev = st
    count = {}
    for st in visit:
        if st:
            count[st] = count.get(st, 0) + 1
    out, seen, lab = [], {}, ''
    for it, v in zip(items, visit):
        st = it[4]
        if v is not None or not st:
            lab = st
            if st and count.get(st, 0) > 1:
                seen[st] = seen.get(st, 0) + 1
                lab = st + 'abcdefghijklmnopqrstuvwxyz'[seen[st] - 1]
        out.append(it[:4] + (lab,))
    return out


def tree_items(ev):
    """(colour, [lines], answer, flags, step) for the nodes of the tree."""
    items, pending = [], []
    phase, early = None, False
    cut = False     # the last pruning check of this screening removed spectra
    for e in ev:
        before = len(items)
        n, q, a = e['node'], e['question'], e['answer']
        if n == 'fit' and q == f"fit of {e.get('model')}":
            phase = None
        if n == 'resid':
            phase = 'wr' if q.startswith('Res_wr') else 'r0'
        if n == 'r1sigma':
            early = a == 'Yes'
        if n == 'r3sigma':
            early = False
        if n not in ('fit', 'prune_check'):
            cut = False
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
                s = f"  {fam_tex(g)}: {p['answer']}"
                if p.get('stage') == 'lrt':
                    thr = p.get('dchi2_thr')
                    s += (f"  (Δχ² = {p['dchi2']:.1f}"
                          + (f" vs {thr:.1f}" if thr is not None
                             and thr == thr else '')
                          + f", p = {p['pvalue']:.2g})")
                elif p['answer'] == 'detected' and p.get('logZ') is not None:
                    s += f"  (residual ln Z = {p['logZ']:.1f})"
                lines.append(s)
            pending = []
            items.append(('purple', lines + ['moment detected?'], a))
        elif n in ('pref', 'prune'):
            lines = [shorten(q)]
            flags = node_flags(e)
            if e.get('method'):
                lines.append(f"ln B ≈ {e['lnB']:.1f} from the ML fits: "
                             "clearly rejected, no PolyChord")
            elif 'lnB' in e:
                s = f"ln B = {e['lnB']:.2f} ± {e['lnB_err']:.2f}"
                if max(e.get('pc_runs', [1])) > 1:
                    s += f"  (PolyChord runs {e['pc_runs']})"
                lines.append(s)
            items.append(('red', lines, a, flags))
        elif n == 'prune_check':
            r0 = ' (r = 0)' if q.endswith('(r = 0)?') else ' (r free)'
            lab = LABEL.search(q)
            if e.get('undetected'):
                left = short(model_from_label(lab.group(0))
                             .without(e['undetected']).label) if lab else '?'
                if q.startswith('Undetected new'):
                    # screening (ML + emcee): removed right away
                    head, tail = 'removed (0 in their 68% HPD)', 'left: '
                else:
                    # full fit: removed only if the pruned model is not
                    # disfavoured (next box)
                    head, tail = ('full fit, 0 in their 68% HPD (removed if '
                                  'not disfavoured)'), 'pruned model: '
                items.append(('purple', [f'pruning{r0}: {head}',
                                         ', '.join(amp_tex(x) for x in
                                                   e['undetected']),
                                         tail + left], None))
                cut = True
            else:
                # first check of a screening that removes nothing: shown
                # too, so that every pruning step appears in the path
                if q.startswith('Undetected new') and not cut and lab:
                    items.append(('purple', [f'pruning{r0}: nothing removed '
                                             '(all new spectra detected)',
                                             'left: ' + short(lab.group(0))],
                                  None))
                cut = False
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
        st = tree_step(e, phase, early)
        for k in range(before, len(items)):
            col, lines, ans = items[k][:3]
            fl = items[k][3] if len(items[k]) > 3 else []
            if col in ('blue', 'orange'):
                col = 'ul' if col == 'blue' else 'det'
            elif n in QUESTION:
                col = 'dia'
            else:
                col = 'box'
            items[k] = (col, lines, ans, fl, st)
    return letters(items)


# flags as in the results page: PTE < 0.01; '≈' a decision taken on a
# borderline PolyChord comparison (|ln B - threshold| < 2 errors after the
# reruns); 'r=0' a model with r = 0 against M_0,r: preferred by a lot (✓✓),
# by a little (✓, |ln B| < max(LNB_FLAG, 2 errors)), not preferred by a
# little (✗); ≈ = choice with |ln B - threshold| < max(LNB_FLAG, 2 errors)
# (LNB_FLAG = 1: inconclusive on the Jeffreys scale of Trotta 2008, the
# lnB_flag of the tree)
PTE_FLAG = 0.01
UND_BAND = 2.
LNB_FLAG = 1.


def band(err):
    """Half width of the 'by a little' band around a threshold."""
    return max(LNB_FLAG, UND_BAND * (err if err == err else 0.))


def r0_flag(x, err):
    """x = ln B(model with r = 0 vs M_0,r)."""
    if x > 0:
        return 'r=0 ✓✓' if x >= band(err) else 'r=0 ✓'
    return 'r=0 ✗' if x > -band(err) else None


def node_flags(e):
    """Flags of a comparison node ('pref' or 'prune')."""
    if 'lnB' not in e or e.get('method'):
        return []
    thr = e.get('lnB_preferred', e.get('lnB_prune', 0.))
    out = ['≈'] if abs(e['lnB'] - thr) < band(e['lnB_err']) else []
    if e['node'] == 'pref' and e['question'].startswith('Is M_1,0'):
        f = r0_flag(e['lnB'], e['lnB_err'])
        out += [f] if f else []
    return out


def end_item(e, model, chi=None, where=''):
    """Conclusion box; chi = (chi2, dof) of the final model."""
    extra = [] if chi is None else \
        [f"chi2{where} = {chi[0]:.1f} / {chi[1]},  "
         f"PTE = {stats.chi2.sf(chi[0], chi[1]):.3g}"]
    if e.get('low_pte'):
        extra.append("low PTE: the model does not describe the data well")
    if e.get('lnB_rfree_vs_r0') is not None:
        extra.append(f"ln B(r free vs r = 0) = {e['lnB_rfree_vs_r0']:.2f} ± "
                     f"{e['lnB_err']:.2f}  (same model with r = 0; not used "
                     "for the decision)")
    # choices taken with |ln B| < max(1, 2 errors) (also after the
    # repetitions)
    for c in e.get('undecided_choices') or []:
        extra.append(f"undecided: {short(c['chosen'])} over "
                     f"{short(c['other'])} (ln B = {c['lnB']:.2f} ± "
                     f"{c['lnB_err']:.2f})")
    flags = []
    if chi is not None and stats.chi2.sf(chi[0], chi[1]) < PTE_FLAG:
        flags.append('PTE')
    if e.get('lnB_rfree_vs_r0') is not None:
        f = r0_flag(-e['lnB_rfree_vs_r0'], e['lnB_err'])
        flags += [f] if f else []
    if e.get('conclusion') == 'upper_limit':
        r95 = e.get('r_95', e.get('r_95_pl0'))
        return ('blue', ['UPPER LIMIT', model, f"r < {r95:.4f} (95%)"]
                + extra, None, flags)
    if e.get('conclusion') == 'r_detection':
        med = e.get('r_med', e.get('r_med_pl'))
        lo = e.get('r_lo1', e.get('r_lo1_pl'))
        hi = e.get('r_hi1', e.get('r_hi1_pl'))
        return ('orange', ['r DETECTION', model,
                           f"r = {med:.4f} [{lo:.4f}, {hi:.4f}] (68%)"]
                + extra, None, flags)
    return ('grey', [shorten(e['question'])], e['answer'], flags)


def profile_items(ev, res, ndata=None, tree_ev=()):
    """(colour, [lines], answer) for the nodes of the profile scheme;
    ndata: size of the data vector (for the dof at r = r_min); tree_ev: log
    of the tree (families of the Res_r0 chain and their pruned spectra)."""
    comp = (res or {}).get('comparison') or {}
    items = []
    if res:
        cands = '; '.join(fam_left(g, off) for g, off in
                          chain_candidates(list(tree_ev))) or 'none'
        items.append(('pbox', [f"M* = {short(res['model_tree'])}",
                               f"candidates from the last Res_r0: {cands}"],
                      None, [], '20'))
    rows = [e for e in ev if e['node'] == 'compare']
    if rows:
        best = max((e for e in rows if 'logZ' in e),
                   key=lambda e: comp.get(e['question'].split()[0], e)
                   ['logZ'], default=None)
        lines = ['profile in r of each model, then compare each model at its '
                 'own r* (local evidence)']
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
        items.append(('pbox', lines, None,
                      ['≈'] if (res or {}).get('undecided') else [], '21–22'))
    for e in ev:
        n, q, a = e['node'], e['question'], e['answer']
        if n == 'final':
            items.append(('pbox', ['winning model: ' + short(a)], None, [],
                          '22'))
        elif n == 'post':
            items.append(('pbox', ['posterior of r from the profile; other '
                                   'parameters from P(θ | r = r*, data), '
                                   'emcee with r fixed at r*'], None, [],
                          '23'))
        elif n == 'r3sigma':
            items.append(('dia', ['is 0 within the central 3 sigma of P(r)?',
                                  '(posterior from the profile)',
                                  f"99.7%: [{e['r_lo3']:.4f}, "
                                  f"{e['r_hi3']:.4f}]"], a, [], '24'))
        elif n == 'END':
            # chi2 of the final model at the minimum of its profile; free
            # parameters = those of the posterior at fixed r ('post' node)
            post = [p for p in ev if p['node'] == 'post']
            npar = len([k for k in post[-1] if k.endswith('_med')]) \
                if post else None
            chi = (e['chi2_at_rstar'], ndata - npar) \
                if ndata and npar and 'chi2_at_rstar' in e else None
            col, lines, ans, fl = end_item(e, short(e.get('model_final')),
                                           chi, ' at r_min')
            items.append(({'blue': 'ul', 'orange': 'det'}.get(col, col),
                          lines, ans, fl, '24'))
        elif n != 'compare':
            items.append(('grey', [shorten(q)], a, [], ''))
    return letters(items)


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


FLAG_COLORS = {'≈': ('#f2c94c', '#b8901a'), 'PTE': ('#f6b26b', '#b45f06'),
               'r=0': ('#b4a7d6', '#674ea7')}


def draw_column(ax, items, x0, title, num_color='#2a6fdb'):
    """Boxes from the top (y = 0) downwards; returns the boxes (x, y, h)."""
    ax.text(x0 + WIDTH / 2, 1.2, title, ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    y, boxes = 0., []
    for item in items:
        col, lines, ans = item[:3]
        flags = item[3] if len(item) > 3 else []
        wl = wrap(lines)
        if ans is not None:
            wl += [(t, True) for t, _ in wrap([f"→ {shorten(str(ans))}"])]
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
        step = item[4] if len(item) > 4 else ''
        if step:
            ax.text(x0, y, step, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.8',
                              fc=num_color, ec='none'))
        xb = x0 + WIDTH - 0.6
        for fl in reversed(flags):
            fc, ec = FLAG_COLORS.get(fl.split()[0], FLAG_COLORS['≈'])
            t = ax.text(xb, y, fl, ha='right', va='center', fontsize=8.5,
                        fontweight='bold', color='#1b1b1b', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6',
                                  fc=fc, ec=ec, lw=0.8))
            # next badge to the left of this one (its width from the renderer)
            bb = t.get_window_extent(ax.figure.canvas.get_renderer())
            x_left = ax.transData.inverted().transform((bb.x0, bb.y0))[0]
            xb = x_left - 0.7
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
        columns.append((profile_items(pev, res, ndata, ev),
                        'profile scheme'))

    def height(items):
        return sum(len(wrap(it[1])) + (it[2] is not None) for it in items) \
            * LH + len(items) * (2 * PAD + GAP)

    H = max(height(it) for it, _ in columns) + 4
    W = len(columns) * (WIDTH + 6)
    fig = plt.figure(figsize=(W * 0.24, H * 0.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-2, W - 2)
    ax.set_ylim(-H + 3, 4)
    ax.axis('off')

    cols = [draw_column(ax, it, i * (WIDTH + 6), t,
                        '#2a6fdb' if i == 0 else '#7b4fd6')
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
