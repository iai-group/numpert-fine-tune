# Rebuttal significance analysis

## Verification gate (recomputed vs published table)

dataset       model                 original        T1        T2        T3   overall  status
averitec      baseline                 93.90     95.65     95.09     96.83     95.31  PASS
averitec      no_approx                89.08     89.72     93.73     97.20     92.13  PASS
averitec      young_nightingale        91.60     90.12     95.20     97.20     93.20  PASS
spanish-test  baseline                 96.37     95.51     97.17     97.40     96.44  PASS
spanish-test  no_approx                96.66     94.83     99.64     99.54     97.48  PASS
spanish-test  young_nightingale        96.32     94.71     99.62     99.44     97.36  PASS

## averitec: published baseline row reproduction attempts

variant                       original        T1        T2        T3   overall  max|diff|
current_run (May 25):valid_only     93.90     95.65     95.09     96.83     95.31      0.00
current_run (May 25):fail_false     79.83     79.84     97.05     98.13     88.67     15.81
current_run (May 25):fail_true     80.67     78.26     57.20     57.01     68.00     39.82
current_run (May 25):fail_error     64.71     60.87     57.20     57.01     59.60     39.82
earlier_run (May 23):valid_only     96.05     91.39     95.83     98.48     94.79      4.26
earlier_run (May 23):fail_false     80.67     79.05     97.42     99.07     88.80     16.60
earlier_run (May 23):fail_true     78.15     70.36     59.41     60.75     66.27     36.08
earlier_run (May 23):fail_error     61.34     54.55     59.41     60.75     58.27     41.10
published row                    93.90     95.65     95.09     96.83     95.31

## averitec: M_Base (zero-shot) vs M_All-Perturb

### original  (n=119, baseline parseable=82, fine-tuned parseable=119, both=82)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           93.90     91.60     -2.31     [-9.16, +4.59]         -          -
paired_valid        93.90     92.68     -1.22     [-8.33, +5.95]       5/4     1.0000
fail_as_error       64.71     91.60    +26.89   [+17.65, +36.13]      5/37    4.4e-07

### T1  (n=253, baseline parseable=161, fine-tuned parseable=253, both=161)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.65     90.12     -5.53    [-10.03, -0.91]         -          -
paired_valid        95.65     94.41     -1.24     [-5.56, +3.07]       7/5     0.7744
fail_as_error       60.87     90.12    +29.25   [+23.32, +35.57]      7/81    4.5e-17

### T2  (n=271, baseline parseable=163, fine-tuned parseable=271, both=163)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.09     95.20     +0.11     [-3.39, +3.87]         -          -
paired_valid        95.09     93.87     -1.23     [-5.42, +2.99]       7/5     0.7744
fail_as_error       57.20     95.20    +38.01   [+31.73, +44.28]     7/110    6.4e-25

### T3  (n=107, baseline parseable=63, fine-tuned parseable=107, both=63)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           96.83     97.20     +0.37     [-4.67, +6.10]         -          -
paired_valid        96.83     96.83     +0.00     [-6.35, +6.25]       2/2     1.0000
fail_as_error       57.01     97.20    +40.19   [+29.91, +50.47]      2/45    1.6e-11

### overall  (n=750, baseline parseable=469, fine-tuned parseable=750, both=469)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.31     93.20     -2.11     [-4.51, +0.33]         -          -
paired_valid        95.31     94.24     -1.07     [-3.60, +1.46]     21/16     0.5114
fail_as_error       59.60     93.20    +33.60   [+29.87, +37.33]    21/273    4.4e-57
macro-F1 (published): base=92.17 ft=89.29 delta=-2.87pp CI=[-6.73, +1.09]
macro-F1 (fail_as_error): base=52.09 ft=89.29 delta=+37.21pp CI=[+33.07, +41.30]

## averitec: M_Base (zero-shot) vs M_Excl-Approx

### original  (n=119, baseline parseable=82, fine-tuned parseable=119, both=82)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           93.90     89.08     -4.83    [-12.10, +2.47]         -          -
paired_valid        93.90     92.68     -1.22     [-8.33, +5.88]       5/4     1.0000
fail_as_error       64.71     89.08    +24.37   [+15.13, +33.61]      5/34    2.4e-06

### T1  (n=253, baseline parseable=161, fine-tuned parseable=253, both=161)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.65     89.72     -5.93    [-10.69, -1.23]         -          -
paired_valid        95.65     95.03     -0.62     [-5.00, +3.80]       7/6     1.0000
fail_as_error       60.87     89.72    +28.85   [+22.53, +35.18]      7/80    8.3e-17

### T2  (n=271, baseline parseable=163, fine-tuned parseable=271, both=163)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.09     93.73     -1.37     [-4.71, +2.17]         -          -
paired_valid        95.09     91.41     -3.68     [-7.84, +0.59]       9/3     0.1460
fail_as_error       57.20     93.73    +36.53   [+29.89, +42.80]     9/108    1.1e-22

### T3  (n=107, baseline parseable=63, fine-tuned parseable=107, both=63)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           96.83     97.20     +0.37     [-4.67, +6.12]         -          -
paired_valid        96.83     95.24     -1.59     [-8.77, +5.17]       3/2     1.0000
fail_as_error       57.01     97.20    +40.19   [+29.91, +50.47]      3/46    7.0e-11

### overall  (n=750, baseline parseable=469, fine-tuned parseable=750, both=469)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.31     92.13     -3.18     [-5.64, -0.70]         -          -
paired_valid        95.31     93.39     -1.92     [-4.50, +0.66]     24/15     0.1996
fail_as_error       59.60     92.13    +32.53   [+28.80, +36.27]    24/268    2.5e-53
macro-F1 (published): base=92.17 ft=87.80 delta=-4.37pp CI=[-8.24, -0.42]
macro-F1 (fail_as_error): base=52.09 ft=87.80 delta=+35.71pp CI=[+31.56, +39.80]

## spanish-test: M_Base (zero-shot) vs M_All-Perturb

### original  (n=2633, baseline parseable=1789, fine-tuned parseable=2633, both=1789)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           96.37     96.32     -0.05     [-1.04, +0.93]         -          -
paired_valid        96.37     98.32     +1.96     [+1.18, +2.78]     10/45    2.1e-06
fail_as_error       65.48     96.32    +30.84   [+29.09, +32.70]    10/822   2.9e-228

### T1  (n=5725, baseline parseable=3299, fine-tuned parseable=5725, both=3299)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.51     94.71     -0.81     [-1.60, +0.00]         -          -
paired_valid        95.51     97.82     +2.30     [+1.66, +2.97]     22/98    1.2e-12
fail_as_error       55.04     94.71    +39.67   [+38.38, +40.96]   22/2293    0.0e+00

### T2  (n=6131, baseline parseable=2963, fine-tuned parseable=6131, both=2963)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           97.17     99.62     +2.46     [+1.86, +3.09]         -          -
paired_valid        97.17     99.80     +2.63     [+2.05, +3.23]      2/80    1.4e-21
fail_as_error       46.96     99.62    +52.67   [+51.43, +53.94]    2/3231    0.0e+00

### T3  (n=1971, baseline parseable=1039, fine-tuned parseable=1971, both=1039)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           97.40     99.44     +2.04     [+1.13, +3.01]         -          -
paired_valid        97.40     99.62     +2.21     [+1.34, +3.14]      0/23    2.4e-07
fail_as_error       51.34     99.44    +48.10   [+45.92, +50.28]     0/948   8.4e-286

### overall  (n=16460, baseline parseable=9090, fine-tuned parseable=16460, both=9090)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           96.44     97.36     +0.93     [+0.53, +1.35]         -          -
paired_valid        96.44     98.77     +2.33     [+1.99, +2.69]    34/246    8.0e-41
fail_as_error       53.26     97.36    +44.11   [+43.34, +44.86]   34/7294    0.0e+00
macro-F1 (published): base=76.88 ft=85.31 delta=+8.43pp CI=[+6.26, +10.72]
macro-F1 (fail_as_error): base=37.01 ft=85.31 delta=+48.30pp CI=[+47.02, +49.57]

## spanish-test: M_Base (zero-shot) vs M_Excl-Approx

### original  (n=2633, baseline parseable=1789, fine-tuned parseable=2633, both=1789)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           96.37     96.66     +0.29     [-0.64, +1.26]         -          -
paired_valid        96.37     98.32     +1.96     [+1.18, +2.77]      9/44    1.2e-06
fail_as_error       65.48     96.66    +31.18   [+29.40, +33.04]     9/830   3.0e-232

### T1  (n=5725, baseline parseable=3299, fine-tuned parseable=5725, both=3299)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           95.51     94.83     -0.68     [-1.46, +0.11]         -          -
paired_valid        95.51     97.79     +2.27     [+1.64, +2.92]     20/95    7.1e-13
fail_as_error       55.04     94.83    +39.79   [+38.53, +41.08]   20/2298    0.0e+00

### T2  (n=6131, baseline parseable=2963, fine-tuned parseable=6131, both=2963)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           97.17     99.64     +2.48     [+1.88, +3.10]         -          -
paired_valid        97.17     99.73     +2.56     [+1.98, +3.18]      4/80    2.1e-19
fail_as_error       46.96     99.64    +52.68   [+51.44, +53.97]    4/3234    0.0e+00

### T3  (n=1971, baseline parseable=1039, fine-tuned parseable=1971, both=1039)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           97.40     99.54     +2.14     [+1.22, +3.12]         -          -
paired_valid        97.40     99.62     +2.21     [+1.30, +3.17]      1/24    1.5e-06
fail_as_error       51.34     99.54    +48.20   [+45.97, +50.38]     1/951   5.0e-284

### overall  (n=16460, baseline parseable=9090, fine-tuned parseable=16460, both=9090)
protocol         acc_base    acc_ft  delta_pp        95% CI (pp)       b/c  McNemar p
published           96.44     97.48     +1.04     [+0.65, +1.45]         -          -
paired_valid        96.44     98.73     +2.30     [+1.95, +2.66]    34/243    4.3e-40
fail_as_error       53.26     97.48    +44.22   [+43.46, +44.98]   34/7313    0.0e+00
macro-F1 (published): base=76.88 ft=85.89 delta=+9.01pp CI=[+6.89, +11.24]
macro-F1 (fail_as_error): base=37.01 ft=85.89 delta=+48.88pp CI=[+47.61, +50.14]
