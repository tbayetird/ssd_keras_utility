from utils import generateconfig as genc
import os
import gc
## Generating configuration
genc.setup_config('D:\\datas\\pirogues-mer\\test','pirogues_mer_test')


##Training
cmd = 'python {}'.format(os.path.dirname(os.path.realpath(__file__))+'\\train.py')
cmd += ' -n {}'.format('pirogues_mer_test')
os.system(cmd)
gc.collect()

## Testing
cmd = 'python {}'.format(os.path.dirname(os.path.realpath(__file__))+'\\test.py')
cmd += ' -n {}'.format('pirogues_mer_test')
os.system(cmd)
