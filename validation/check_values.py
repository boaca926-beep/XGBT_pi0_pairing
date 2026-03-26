## Event-wise
nb_evnt_val = 220463 # total number of test events

## Confusion matrix
tp = 141385 # false-negative 
fp = 31035 # false-positive
tn = 11450 # false-negative
fn = 16349 # true-positive
cm = tn + fp + fn + tp
accuracy = (tn + tp) / cm
precision = tp / (fp + tp)
recall = tp / (fn + tp)
F1 = 2 * (precision * recall) / (precision + recall)

print(f"Total number of validation events: {nb_evnt_val}, pairs: {nb_evnt_val}\n")
#print(f"    1: {nb_evnt_val_pair_good} ({nb_evnt_val_pair_good/nb_evnt_val_pair*100:0.1f}%) 0: {nb_evnt_val_pair_bad} ({nb_evnt_val_pair_bad/nb_evnt_val_pair*100:0.1f}%)")
#print(f"    1+0: {nb_evnt_val_pair_bad + nb_evnt_val_pair_good}\n")
print(f"    tn: {tn} fp: {fp}")
print(f"    fn: {fn} tp: {tp}")
print(f"cm entry: {cm}")
print(f"accuracy = {accuracy* 100:0.1f}%")
print(f"precision = {precision* 100:0.1f}%")
print(f"recall = {recall* 100:0.1f}%")
print(f"F1 = {F1* 100:0.1f}%") # Harmonic Mean = 2 / (1/a + 1/b) = 2ab / (a + b)



# Pair-wise
nb_evnt_val_pair = 661389 # total number of test pairs
nb_evnt_val_pair_good = 170157 # true signal events (kloe selection)
nb_evnt_val_pair_bad = 491232 # background events (kloe selection)

## Confusion matrix
tn_pair = 449087 # false-negative 
fp_pair = 42145 # false-positive
fn_pair = 21252 # false-negative
tp_pair = 148905 # true-positive
cm_pair_entry = tn_pair + fp_pair + fn_pair + tp_pair


print(f"Total number of validation events: {nb_evnt_val}, pairs: {nb_evnt_val_pair}\n")
print(f"Number of pairs per events: {nb_evnt_val_pair/nb_evnt_val}")
print(f"    1: {nb_evnt_val_pair_good} ({nb_evnt_val_pair_good/nb_evnt_val_pair*100:0.1f}%) 0: {nb_evnt_val_pair_bad} ({nb_evnt_val_pair_bad/nb_evnt_val_pair*100:0.1f}%)")
print(f"    1+0: {nb_evnt_val_pair_bad + nb_evnt_val_pair_good}\n")
print(f"    tn: {tn_pair} fp: {fp_pair}")
print(f"    fn: {fn_pair} tp: {tp_pair}")
print(f"cm entry: {cm_pair_entry}")