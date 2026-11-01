# Validazione del decision tree su Perlmutter — nside 128, senza modi E

Stesso piano della validazione a nside 64 (`../validation`): 100 realizzazioni di CMB + rumore (seed
7000-7099), r log-uniforme in [1e-3, 0.03] (10 con r = 0), ognuna analizzata con d9s4 e con un cielo
PySM a caso tra d9/d10/d12 x s4/s5/s7 (escluso d9s4). `make_plan.py` e' deterministico: set, seed e r
sono gli stessi delle validazioni v1 e v2 a nside 64 (ma a nside 128 CMB e rumore sono realizzazioni diverse).
Va installato in `PySM_rnd2/` al posto della v2 (risultati della v2 salvati sul Mac in
`PySM_tests/validation_results/v2_nside64_lmax200/`).

Cosa cambia rispetto alla v2 (nside 64):

| | nside 64 (v2) | nside 128 (questo pacchetto) |
|---|---|---|
| mappe | nside 64 | **nside 128** |
| bandpower, fit | bin da 10 fino a 192, `l_max` 200 | bin da 10 fino a 384, **`l_max` 250** |
| PySM | direttamente a nside 64 (media dei pixel: finestra del pixel e aliasing nei foreground) | a nside 512, poi **downgrade armonico** a 128 (band-limited come CMB e rumore); mappe gia' pronte in `data/pysm_cache` |
| modi E | si' | **no** (`--no-ee`): la maschera pesata con gli hit + purificazione disperde E in B, cosa che nel modello non c'e' e che nella pipeline vera correggono altri |
| maschera, hit | nside 64 | nside 128 (dalla mappa degli hit di SO a nside 512, ruotata in galattiche a 512) |
| covarianza | 500 sims gaussiane d1s1 | 500 sims gaussiane d1s1 a nside 128, senza E, parametri rifittati su d1s1 con il downgrade armonico (`cov_d1s1_hits_pg_n128_noee.tar`) |
| fiducial HL | `cls_fid_d1s1_hits.fits` | `cls_fid_d1s1_hits_pg_n128.fits` |

Gli r veri stanno in `truth/r_true.csv`: non aprirlo fino alla fine.

## Una volta (nodo di login)

Dal Mac, copia il pacchetto e la covarianza:

```bash
cd ~/Workspace/SimonsObs/BBpower_RT/nersc
tar -cf - validation_n128 cov_d1s1_hits_pg_n128_noee.tar \
  | ssh krach@perlmutter.nersc.gov "tar -C /pscratch/sd/k/krach/SObs/BBpower_Bsure -xf -"
```

Poi al NERSC:

```bash
source /global/homes/k/krach/usr/activate_bbpower.sh
cd /pscratch/sd/k/krach/SObs/BBpower_Bsure
bash validation_n128/code/setup.sh /pscratch/sd/k/krach/SObs/BBpower_Bsure/PySM_rnd2 \
     /pscratch/sd/k/krach/SObs/BBpower_Bsure/cov_d1s1_hits_pg_n128_noee.tar
```

## Lancio

```bash
cd /pscratch/sd/k/krach/SObs/BBpower_Bsure/PySM_rnd2
sbatch -A mp107b -t 03:00:00 code/job_validation.slurm 0 7
sbatch -A mp107d -t 03:00:00 code/job_validation.slurm 8 15
```

Un nodo fa 16 tree insieme (8 set x 2 cieli). I set gia' conclusi vengono saltati e quelli interrotti
ripartono dai fit gia' fatti: se il job finisce il tempo basta rilanciare lo stesso comando.

## Durante e dopo

```bash
squeue -u krach                       # stato del job
sacct -u krach -S <data> -o jobid,jobname%15,account,state,elapsed   # anche i job finiti
tail logs/slurm_<jobid>.out           # set conclusi / falliti
tail logs/set_003_rand.log            # un tree
python code/collect.py $PWD           # tabella alla cieca -> results.csv
python code/collect.py $PWD --unblind # SOLO alla fine: confronto con gli r veri
```

## Struttura di `PySM_rnd2/`

| | |
|---|---|
| `code/`, `data/`, `cov/` | codice, dati, spettri delle sims della covarianza |
| `plan/sets.csv` | indice, seed, cielo casuale |
| `truth/r_true.csv` | r veri (non aprire) |
| `runs/set_<i>_<d9s4\|rand>/` | spettri (`output/`), tree (`tree/decision_log.txt`, `conclusion.json`) |
| `logs/` | log di ogni set e dei job |
