#!/usr/bin/env python

"a more polished example of using hyperband"
"includes displaying best results and saving to a file"

import sys
#import cPickle as pickle
import pickle
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from hyperband import Hyperband

#from defs.gb import get_params, try_params
#from defs.rf import get_params, try_params
#from defs.xt import get_params, try_params
#from defs.rf_xt import get_params, try_params
#from defs.sgd import get_params, try_params
from defs.keras_mlp import get_params, try_params
#from defs.polylearn_fm import get_params, try_params
#from defs.polylearn_pn import get_params, try_params
#from defs.xgb import get_params, try_params
#from defs.meta import get_params, try_params


#--------------------------------------------------random search
try:
	output_file_RS = sys.argv[1]
	if not output_file_RS.endswith( '.pkl' ):
		output_file_RS += '.pkl'	
except IndexError:
	output_file_RS = 'results_RS.pkl'
	
print("RS Will save results to", output_file_RS)

hb2 = Hyperband( get_params, try_params )
results_RS = hb2.run2( skip_last = 0)


print("{} total, best:\n".format(len(results_RS)))

for r in sorted( results_RS, key = lambda x: x['loss'] )[:10]:
	print( "loss: {:.2%} |auc: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
		r['loss'],r['auc'], r['seconds'], r['iterations'], r['counter'] ))
	pprint( r['params'] )
	print

print ("saving...")

with open( output_file_RS, 'wb' ) as f:
	pickle.dump( results_RS, f )
   
staRS_loss=[sub['best_loss'] for sub in results_RS]
staRS_sec=[sub['seconds'] for sub in results_RS]

'''
#-------------------------------------------------- hyperband
try:
	output_file = sys.argv[1]
	if not output_file.endswith( '.pkl' ):
		output_file += '.pkl'	
except IndexError:
	output_file = 'results_2.pkl'
	
print("HB Will save results to", output_file)


hb = Hyperband( get_params, try_params )
results = hb.run( skip_last = 0)

print("{} total, best:\n".format( len( results )))

for r in sorted( results, key = lambda x: x['loss'] )[:10]:
	print( "loss: {:.2%} |auc: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
		r['loss'],r['auc'], r['seconds'], r['iterations'], r['counter'] ))
	pprint( r['params'] )
	print

print ("saving...")

with open( output_file, 'wb' ) as f:
	pickle.dump( results, f )

sta_loss=[sub['best_loss'] for sub in results]
sta_sec=[sub['seconds'] for sub in results]    

#--------------------------------------------------
'''