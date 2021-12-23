#!/usr/bin/env python

"a more polished example of using hyperband"
"includes displaying best results and saving to a file"

import sys
#import cPickle as pickle
import pickle
from pprint import pprint
import numpy as np

from hyperband import Hyperband, Hyperband_LDA, Hyperband_LDA_Iter
# from Embeddings.GLOVE import load_GLOVE

from lda import get_params, try_params

'''
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

'''
#-------------------------------------------------- hyperband / iteration-based
try:
    output_file = sys.argv[1]
    if not output_file.endswith( '.pkl' ):
        output_file += '.pkl'   
except IndexError:
    output_file = 'results_HB.pkl'
    
print("HB Will save results to", output_file)

hb = Hyperband(get_params, try_params)
results = hb.run(skip_last = 0)

print("{} total, best:\n".format(len(results)))

for r in sorted( results, key = lambda x: x['loss'] )[:10]:
    print( "loss: {:.2%} |auc: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
                 r['loss'], r['auc'], r['seconds'], r['iterations'], r['counter']))
    pprint(r['params'])
    print()

print ("saving results...")

with open(output_file, 'wb') as f:
    pickle.dump(results, f)

sta_loss=[sub['best_loss'] for sub in results]
sta_sec=[sub['seconds'] for sub in results]

score = sta_loss
runtime = sta_sec
with open('score_vs_time.csv','w') as f_score:
    f_score.write('Score,Time(s)\n')
    for i in range(len(score)):
        f_score.write('{},{}'.format(score[i], runtime[i]))
'''

#-------------------------------------------------- hyperband for LDA

# $ python main.py results_hb_W2V.pkl W2V

try:
    output_file = sys.argv[1]
    if not output_file.endswith( '.pkl' ):
        output_file += '.pkl'   
except IndexError:
    output_file = 'results_hb_lda.pkl'
    
print("HB will save results to", output_file)

emb_model = sys.argv[2]
print('emb_model =', emb_model)

# if emb_model == 'GLOVE':
#     from Embeddings.GLOVE import load_GLOVE
    
hb = Hyperband_LDA_Iter(get_params, try_params, emb_model)
# results = hb.run(dry_run=True)
results = hb.run()

print ("saving results ...")
with open(output_file, 'wb') as f:
    pickle.dump(results, f)
    

# show the best 10 configs
print("\n---------------- {} total, best 10 are:\n".format(len(results)))

for r in sorted(results, key = lambda x: x['score'], reverse=True)[:10]: # from low to high
    print( "lda_score: {:.4} | {} seconds | {:.1f} n_iter | run {} ".format( 
                 r['score'], r['config_seconds'], r['iterations'], r['counter']))
    
    pprint(r['params'])
    pprint(r['topic_keywords'])
    print()   


# sta_loss=[sub['best_score'] for sub in results]
# sta_sec=[sub['seconds'] for sub in results]

# score = sta_loss
# runtime = sta_sec
# print('saving score_vs_time.csv ...')
# with open('score_vs_time.csv','w') as f_score:
#     f_score.write('score,Time(s)\n')
#     for i in range(len(score)):
#         f_score.write('{},{}\n'.format(score[i], runtime[i]))

print("\nAuto-LDA finished.")

