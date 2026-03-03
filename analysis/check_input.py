#python -c "
import uproot
try:
    file = uproot.open('../data/ksl_sample.root')
    print('All keys:', file.keys())
    print('Classes:', file.classnames())
except Exception as e:
    print('Error:', e)