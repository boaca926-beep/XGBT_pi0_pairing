## Event-wise
nb_evnt_val = 220463 # total number of validation events

## Confusion matrix
tn = 141385 # false-negative 
fp = 31035 # false-positive
fn = 11450 # false-negative
tp = 16349 # true-positive
cm = tn + fp + fn + tp

print(f"Total number of validation events: {nb_evnt_val}, pairs: {nb_evnt_val}\n")
#print(f"    1: {nb_evnt_val_pair_good} ({nb_evnt_val_pair_good/nb_evnt_val_pair*100:0.1f}%) 0: {nb_evnt_val_pair_bad} ({nb_evnt_val_pair_bad/nb_evnt_val_pair*100:0.1f}%)")
#print(f"    1+0: {nb_evnt_val_pair_bad + nb_evnt_val_pair_good}\n")
print(f"    tn: {tn} fp: {fp}")
print(f"    fn: {fn} tp: {tp}")
print(f"cm entry: {cm}")

# Pair-wise
nb_evnt_val_pair = 661389 # total number of validation pairs
nb_evnt_val_pair_good = 170157 # true signal events (kloe selection)
nb_evnt_val_pair_bad = 491232 # background events (kloe selection)

## Confusion matrix
tn_pair = 449416 # false-negative 
fp_pair = 42301 # false-positive
fn_pair = 21053 # false-negative
tp_pair = 148619 # true-positive
cm_pair_entry = tn_pair + fp_pair + fn_pair + tp_pair


print(f"Total number of validation events: {nb_evnt_val}, pairs: {nb_evnt_val_pair}\n")
print(f"Number of pairs per events: {nb_evnt_val_pair/nb_evnt_val}")
print(f"    1: {nb_evnt_val_pair_good} ({nb_evnt_val_pair_good/nb_evnt_val_pair*100:0.1f}%) 0: {nb_evnt_val_pair_bad} ({nb_evnt_val_pair_bad/nb_evnt_val_pair*100:0.1f}%)")
print(f"    1+0: {nb_evnt_val_pair_bad + nb_evnt_val_pair_good}\n")
print(f"    tn: {tn_pair} fp: {fp_pair}")
print(f"    fn: {fn_pair} tp: {tp_pair}")
print(f"cm entry: {cm_pair_entry}")