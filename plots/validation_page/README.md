# Results page of a validation run

One HTML page per run of `nersc/job_validation.slurm`: setup, flowchart, how
the pipeline works, priors, main results, summary tables, plots and the table
of all the sets (filters, flags, and for each set its path through the
flowchart, its corner plot and the details of its choices).

## Inputs

The root of the run, copied from NERSC (only the small files are needed):

```
<root>/results_unblinded.csv      scripts/collect.py tree <root> --unblind
<root>/profile_results.csv        scripts/collect.py profile <root> --out ...
<root>/runs/set_*/tree/decision_log.json, conclusion*.json
<root>/runs/set_*/tree/profile/result.json
<root>/corners/set_*.png          optional (corner plots made at NERSC)
```

and a text file with what changes from run to run: the title, the byline and
setup bullets, and the main results (see `runs/aic_2026-11.html`).

## Usage

```bash
bash plots/validation_page/make_page.sh <root> plots/validation_page/runs/<run>.html <out dir>
```

writes `<out dir>/index.html` with `flowchart.png`, `figures/`, `paths/` and
`corners/` next to it. The folder can go as it is on a web server, e.g. the
NERSC portal (`/global/cfs/cdirs/sobs/www/users/<user>/` →
`https://portal.nersc.gov/cfs/sobs/users/<user>/`; every directory needs
`o+x`, the files `o+r`).

## Pieces

| file | does |
|---|---|
| `make_page.sh` | runs all the steps below |
| `tables.py` | `final_table.csv`, `tree_full.csv`, `profile_as_tree.csv` in `<root>`; PTE at r* in `profile_results.csv` |
| `plot_r_sets.py` | r set by set (true r, median, 68%, r₉₅; red: wrong conclusion, yellow: detection with r < 3×10⁻³) |
| `paths.py` | path of every set through the flowchart (`plots/plot_tree_path.py`) |
| `build_page.py`, `summary.py`, `template.html` | the page (`--standalone`: full HTML document) |
| `runs/*.html` | the text of each run |

Also used: `plots/make_profile_table.py`, `plots/plot_rtrue_vs_rest.py`,
`plots/plot_bias_profile.py`, and the flowchart
`docs/flowcharts/step_by_step/tree_steps.png`.

The flags of the table (PTE < 0.01, ≈ borderline PolyChord decision, r=0
✓✓/✓/✗) are defined in `build_page.py` and drawn with the same names in
`plots/plot_tree_path.py` and in the flowchart.
