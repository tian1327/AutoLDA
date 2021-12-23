import pickle
from lda import print_params, LDA_classifier
from show_results import print_topicwords
import sys

"$ python run_configs.py 1 W2V"


def run_config(n_iterations, params, data):

    print('\n'+'--'*10)
    print_params(params)
    
    # run LDA on data
    lda = LDA_classifier(n_iterations, params)
    
    t_w, n_iter, d_t = lda.fit(data)
    
    lda_score = lda.semantic_score(t_w, emb_model)
    # perplexity = lda.score(data) 

    print('lda_score= {}'.format(lda_score))
    print_topicwords(t_w)

    return {'params': params,'lda_score': lda_score, 'topic_keywords': t_w, 'n_iter':n_iter, 'doc_topic_distr': d_t}


if __name__ == '__main__':


    print('loading full data')
    with open('data.pkl','rb') as pk:
        data = pickle.load(pk)

    num = int(sys.argv[1])
    emb_model = sys.argv[2]
    print('num =',num)
    print('emb_model =', emb_model)

    with open( 'best_10_configs_{}.pkl'.format(emb_model), 'rb' ) as f:
        best_10 = pickle.load( f )

    # with open( 'worst_10_configs.pkl', 'rb' ) as f:
    #     worst_10 = pickle.load( f )


    print('\n'+'==='*10+' best {}:'.format(num))
    good_res = []
    for idx, good in enumerate(best_10):
    
        if idx == num:
            break
        print('idx =',idx)
        res = run_config(81, good['params'], data)
        good_res.append(res)


    # print('\n'+'==='*10+' worst {}:'.format(num))
    # bad_res = []
    # for idx, bad in enumerate(worst_10):

    #     if idx == num:
    #         break
    #     print('idx =',idx)
    #     res = run_config(81, bad['params'], data)
    #     bad_res.append(res)

    print ("saving best res ......")
    with open('best_{}_configs_res_{}.pkl'.format(num, emb_model), 'wb') as f:
        pickle.dump(good_res, f)

    # print ("saving worst res ......")
    # with open('worst_{}_configs_res.pkl'.format(num), 'wb') as f:
    #     pickle.dump(bad_res, f)

    # save the topic_distri in good_res
    with open('doc_topic_distr_{}.csv'.format(emb_model),'w') as f:
        # f.write('doc_id,topic1,topic2,topic3,topic4,topic5\n')
        d_t = good_res[0]['doc_topic_distr'] # take the first, assume the best 
        for r in range(d_t.shape[0]):
            for c in range(d_t.shape[1]):
                f.write('{}\t'.format(d_t[r,c]))
            f.write('\n')

