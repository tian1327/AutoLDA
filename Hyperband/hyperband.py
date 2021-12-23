import numpy as np
from random import random
from math import log, ceil
from time import time, ctime


class Hyperband:
    
    def __init__( self, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function
        
        self.max_iter = 81      # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = []    # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        
    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        
        print('\nRunning Hyperband ......')
        start_time = time()
        
        for s in reversed( range( self.s_max + 1 )):
            
            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))    
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** ( -s )        

            # n random configurations
            T = [ self.get_params() for i in range(n)] 
            
            for i in range(( s + 1 ) - int(skip_last)):    # changed from s + 1
                
                # Run each of the n configs for <iterations> 
                # and keep best (n_configs / eta) configurations
                
                n_configs = n * self.eta ** ( -i )
                n_iterations = r * self.eta ** ( i )
                
                print ("\n*** {} configurations x {:.1f} iterations each".format( 
                    n_configs, n_iterations ))
                
                val_losses = []
                early_stops = []
                
                for t in T:
                    
                    self.counter += 1
                    print ("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
                        self.counter, ctime(), self.best_loss, self.best_counter ))
                    
                    print ("\ns={} | ni={} | ri={}\n".format( 
                        s, n_configs, n_iterations))
                    
                    #start_time = time()
                    
                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params( n_iterations, t)        # <--- run configs here +++++++++
                        
                    assert( type(result) == dict )
                    assert( 'loss' in result )
                    
                    seconds = int(round(time() - start_time ))
                    print ("\n{} seconds.".format( seconds ))
                    
                    loss = result['loss']    
                    val_losses.append(loss)
                    
                    early_stop = result.get('early_stop', False )
                    early_stops.append(early_stop)
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter
                    
                    result['best_loss'] = self.best_loss
                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations
                    
                    self.results.append(result)
                
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[ 0:int(n_configs/self.eta )]
        
        return self.results
    
#"""------------------------------------------------------------------------------------Random search 20200520 TL"""
    def run2( self, skip_last = 0, dry_run = False ):
            print('running RS.......\n')
        
            n=self.max_iter # n random configurations
            n=30
            T = [ self.get_params() for i in range(n)] 
            print('n={}'.format(n))
            #print('T={}'.format(T))
            val_losses = []
            early_stops = []
            start_time = time()
            
            for t in T:
                self.counter += 1
                print ("\n{}/{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
                        self.counter,self.max_iter, ctime(), self.best_loss, self.best_counter ))                            
                
                result = self.try_params( self.max_iter, t )        # <--- run configs here +++++++++
                                
                assert( type( result ) == dict )
                assert( 'loss' in result )
                            
                seconds = int( round( time() - start_time ))
                print ("\n{} seconds.".format( seconds ))
                
                loss = result['loss']
                val_losses.append( loss )
                            
                early_stop = result.get( 'early_stop', False )
                early_stops.append( early_stop )
                            
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_counter = self.counter
                            
                result['best_loss'] = self.best_loss
                result['counter'] = self.counter
                result['seconds'] = seconds
                result['params'] = t
                result['iterations'] = self.max_iter
                            
                self.results.append( result )

            return self.results    

# data-based

class Hyperband_LDA:
    
    def __init__(self, get_params_function, try_params_function, max_doc, emb_model):
        self.get_params = get_params_function
        self.try_params = try_params_function
        
        self.max_iter = max_doc #81      # maximum resources allocated per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int( self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter # total budget for all s

        self.results = []    # list of dicts
        self.counter = 0
        self.best_score = 0
        self.best_counter = -1

        self.emb_model = emb_model
        

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        
        print('\nRunning Hyperband ......')
        start_time = time()
        
        for s in reversed( range( self.s_max + 1 )):
            
            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))    
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)        

            # n random configurations
            T = [ self.get_params() for i in range(n)] 
            
            for i in range(( s + 1 ) - int(skip_last)):    # changed from s + 1
                
                # Run each of the n configs for <iterations> 
                # and keep best (n_configs / eta) configurations
                
                n_configs = int(n * self.eta ** ( -i ))
                n_doc = int(r * self.eta ** ( i ))
                # n_doc = max(int(r * self.eta ** ( i )), 200)
                
                print("\n*** {} configurations x {:.1f} documents each".format( 
                    n_configs, n_doc))
                
                lda_scores = []
                # early_stops = []
                
                for t in T:                    
                    self.counter += 1
                    print ("\n{} | {} | best score so far: {:.4f} (run {})".format( 
                        self.counter, ctime(), self.best_score, self.best_counter))
                    
                    print ("s={} | ni={} | ri={}".format(s, n_configs, n_doc))
                    
                    # start_time = time()
                    
                    if dry_run:
                        result = {'lda_score': random()}
                    else:
                        result = self.try_params(n_doc, t, self.emb_model)        # <--- run configs here +++++++++
                        
                    assert(type(result) == dict)
                    assert('lda_score' in result)
                    
                    seconds = round(time() - start_time, 6)
                    print("{} seconds".format(seconds))
                    
                    score = result['lda_score']    
                    lda_scores.append(score)
                    
                    # early_stop = result.get('early_stop', False)
                    # early_stops.append(early_stop)
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if score > self.best_score:
                        self.best_score = score
                        self.best_counter = self.counter
                    
                    result['best_score'] = self.best_score
                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['n_doc'] = n_doc
                    
                    self.results.append(result)
                
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                lda_scores = np.array(lda_scores)
                indices = np.argsort(-lda_scores) # from high to low
                
                T = [T[i] for i in indices]
                T = T[0:int(n_configs/self.eta)]
        
        return self.results


# iteration-based Hyperband with LDA 

# class Hyperband_LDA_Iter:
    
#     def __init__(self, get_params_function, try_params_function, emb_model):
#         self.get_params = get_params_function
#         self.try_params = try_params_function
        
#         self.max_iter = 81      # maximum iterations allocated per configuration
#         self.eta = 3            # defines configuration downsampling rate (default = 3)

#         self.logeta = lambda x: log(x) / log(self.eta)
#         self.s_max = int( self.logeta(self.max_iter))
#         self.B = (self.s_max + 1) * self.max_iter # total budget for all s

#         self.results = []    # list of dicts
#         self.counter = 0
#         self.best_score = 
#         self.best_counter = -1

#         self.emb_model = emb_model
        

#     # can be called multiple times
#     def run(self, skip_last=0, dry_run=False):
        
#         print('\nRunning Hyperband ......')
#         start_time = time()
        
#         for s in reversed( range( self.s_max + 1 )):
            
#             # initial number of configurations
#             n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))    
            
#             # initial number of iterations per config
#             r = self.max_iter * self.eta ** (-s)        

#             # n random configurations
#             T = [ self.get_params() for i in range(n)] 
            
#             for i in range(( s + 1 ) - int(skip_last)):    # changed from s + 1
                
#                 # Run each of the n configs for <iterations> 
#                 # and keep best (n_configs / eta) configurations
                
#                 n_configs = int(n * self.eta ** ( -i ))
#                 n_doc = int(r * self.eta ** ( i ))
#                 # n_doc = max(int(r * self.eta ** ( i )), 200)
                
#                 print("\n*** {} configurations x {:.1f} documents each".format( 
#                     n_configs, n_doc))
                
#                 lda_scores = []
#                 # early_stops = []
                
#                 for t in T:                    
#                     self.counter += 1
#                     print ("\n{} | {} | best score so far: {:.4f} (run {})".format( 
#                         self.counter, ctime(), self.best_score, self.best_counter))
                    
#                     print ("s={} | ni={} | ri={}".format(s, n_configs, n_doc))
                    
#                     # start_time = time()
                    
#                     if dry_run:
#                         result = {'lda_score': random()}
#                     else:
#                         result = self.try_params(n_doc, t, self.emb_model)        # <--- run configs here +++++++++
                        
#                     assert(type(result) == dict)
#                     assert('lda_score' in result)
                    
#                     seconds = round(time() - start_time, 6)
#                     print("{} seconds".format(seconds))
                    
#                     score = result['lda_score']    
#                     lda_scores.append(score)
                    
#                     # early_stop = result.get('early_stop', False)
#                     # early_stops.append(early_stop)
                    
#                     # keeping track of the best result so far (for display only)
#                     # could do it be checking results each time, but hey
#                     if score > self.best_score:
#                         self.best_score = score
#                         self.best_counter = self.counter
                    
#                     result['best_score'] = self.best_score
#                     result['counter'] = self.counter
#                     result['seconds'] = seconds
#                     result['params'] = t
#                     result['n_doc'] = n_doc
                    
#                     self.results.append(result)
                
#                 # select a number of best configurations for the next loop
#                 # filter out early stops, if any
#                 lda_scores = np.array(lda_scores)
#                 indices = np.argsort(-lda_scores) # from high to low
                
#                 T = [T[i] for i in indices]
#                 T = T[0:int(n_configs/self.eta)]
        
#         return self.results

class Hyperband_LDA_Iter:
    
    def __init__( self, get_params_function, try_params_function, emb_model):
        self.get_params = get_params_function
        self.try_params = try_params_function
        
        self.max_iter = 81      # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = []    # list of dicts
        self.counter = 0
        self.best_score = 0
        self.best_counter = -1
        
        self.emb_model = emb_model

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        
        print('\nRunning Hyperband ......')
        start_time = time()
        
        for s in reversed( range( self.s_max + 1 )):
            
            # initial number of configurations
            n = ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s )
            
            print('s = {}, n ={}'.format(s, n))
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** ( -s )        

            # n random configurations
            T = [ self.get_params() for i in range(n)] 
            
            for i in range(( s + 1 ) - int(skip_last)):    # changed from s + 1
                
                # Run each of the n configs for <iterations> 
                # and keep best (n_configs / eta) configurations
                
                n_configs = int(n * self.eta ** ( -i ))
                n_iterations = int(r * self.eta ** ( i ))
                
                print ("\n*** {} configurations x {:.1f} iterations each".format(n_configs, n_iterations))
                
                val_losses = []
                # early_stops = []
                
                print ("s={} | ni={} | ri={}\n".format(s, n_configs, n_iterations))
                
                for t in T:
                    
                    config_start = time()
                    self.counter += 1
                    print ("\n{} | {} | best lda_score so far: {:.4f} (run {})\n".format( 
                        self.counter, ctime(), self.best_score, self.best_counter))
                    
                    
                    #start_time = time()
                    
                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params(n_iterations, t, self.emb_model)        # <--- run configs here +++++++++
                        
                    assert(type(result) == dict)
                    assert('lda_score' in result )
                    
                    config_sec = time() - config_start
                    seconds = time() - start_time
                    
                    print('config_sec = {:.4f} seconds | Cumulative {:.4f} seconds'.format(config_sec, seconds))
                    
                    lda_score = result['lda_score']    
                    val_losses.append(lda_score)
                    
                    # early_stop = result.get('early_stop', False)
                    # early_stops.append(early_stop)
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if lda_score > self.best_score:
                        self.best_score = lda_score
                        self.best_counter = self.counter
                    
                    result['best_score'] = self.best_score
                    result['counter'] = self.counter
                    result['config_seconds'] = config_sec
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations
                    result['score'] = lda_score
                    
                    self.results.append(result)
                
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                # T = [T[i] for i in indices if not early_stops[i]]
                T = [T[i] for i in indices]
                T = T[ 0:int(n_configs/self.eta )]
        
        return self.results