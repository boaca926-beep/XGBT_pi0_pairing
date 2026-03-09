#!/bin/bash

path_header="./path_bdt.h"

echo -e 'const TString model_filename = ' > $path_header
echo -e 'const TString data_filename = ' >> $path_header

model_filename="../training/models/bdt_pi0_TCOMB.root"
data_filename="../data/kloe_small_sample.root"

sed -i 's|\(const TString model_filename =\)\(.*\)|\1 "'"${model_filename}"'";|' "$path_header"
sed -i 's|\(const TString data_filename =\)\(.*\)|\1 "'"${data_filename}"'";|' "$path_header"


