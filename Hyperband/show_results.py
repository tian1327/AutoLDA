#!/usr/bin/env python

"load pickled results, show the best"

"$ python show_results.py results_hb_w2v.pkl 10 W2V"

import sys
import pickle

from pprint import pprint

def print_topicwords(topicwords):
    
    for (i, t) in enumerate(topicwords):
        print('\nTopic {}:'.format(i+1), end=' ')
        for w in t:
            print(w, end=' ')
    print()


def main():
    try:
    	input_file = sys.argv[1]
    except IndexError:
    	print ("Usage: python show_results.py results.pkl [number of results to show] [model]\n")
    	raise SystemExit

    try:
    	results_to_show = int(sys.argv[2])
    except IndexError:
    	results_to_show = 10

    model = sys.argv[3]

    with open( input_file, 'rb' ) as i_f:
    	results = pickle.load( i_f )


    if results_to_show > 0: # from high to low, the higher the better
        print("The best {} configs are:\n".format(results_to_show))
        configs_show = sorted(results, key = lambda x: x['score'], reverse=True)[:results_to_show]

    else:
        print("The worst {} configs are:\n".format(-results_to_show))
        configs_show = sorted(results, key = lambda x: x['score'], reverse=False)[:-results_to_show]

    for r in configs_show: 
        print( "score: {:.4} | {} seconds | {:.1f} n_iter | run {} ".format( 
                     r['score'], r['config_seconds'], r['iterations'], r['counter']))
        
        pprint(r['params'])
        print('topic_words in train_data:')
        print_topicwords(r['topic_keywords'])
        print()    

    # save best 10 configs
    best_10 = sorted(results, key = lambda x: x['score'], reverse=True)[:10]
    print ("saving best_10_configs_{}.pkl ...".format(model))
    with open('best_10_configs_{}.pkl'.format(model), 'wb') as f:
        pickle.dump(best_10, f)

    # save worst 10 configs, pointless/incorrect
    # worst_10 = sorted(results, key = lambda x: x['loss'], reverse=True)[:10]
    # print ("saving worst_10 ......")
    # with open('worst_10_configs.pkl', 'wb') as f:
    #     pickle.dump(worst_10, f)
    
    score = [sub['best_score'] for sub in results]
    runtime = [sub['seconds'] for sub in results]
    
    print('saving score_vs_time_{}.csv ...'.format(model))
    with open('score_vs_time_{}.csv'.format(model),'w') as f_score:
        f_score.write('score,Time(s)\n')
        for i in range(len(score)):
            f_score.write('{},{}\n'.format(score[i], runtime[i]))

if __name__ == '__main__':
    main()