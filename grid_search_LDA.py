import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
import gensim
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import sys
import csv
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
import time
import matplotlib.pyplot as plt
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

import sys
sys.path.append('/home/kexin/Desktop/AutoLDA/Hyperband')
from lda_metric import embedding_distance


np.set_printoptions(threshold= sys.maxsize)
np.random.seed(5)

load_model = False
model_name = 'best_LDA.pkl'
save_to_model = False
save_model_name = 'best_LDA.pkl'

# select from 'coherence', 'perplexity', 'embedding'
# the perplexity takes the negative so that it can be maximized
score_type = 'embedding'
# set the embedding model from ["BERT", "GLOVE", "W2V", "ELMo"] if use embedding
embedding_model = 'GLOVE'

param_dist = {"max_df": list(np.linspace(0.7, 1, 2)),					
			  "min_df": list(np.linspace(0.1, 0.2, 2)),		 
			  "topic_number": list(np.linspace(5, 20, 16, dtype='int')),	
			  "learning_decay": list(np.linspace(0.51, 1, 2)),
			  "learning_offset": list(np.linspace(1, 21, 21, dtype='int')),
			  "batch_size":[16, 32, 64, 128, 256],
			  "max_iter":[100, 200]
			  }


ny_long = ['__aNiRlpOtM','_2C4tWdpdfE','_5Dlspmarck','_dlVkCeP3y4','_DNudv1Rtfg','_GLjAs7rvNo','_mOhVtoBUUg','_Nst1BllAx8','_QJQzD6Qgkk','-1H2myJkHXY','-e0j4Oz_kOg','-klLyOizv8c','-ruy-w0bxvA','-RWCteAeCsU','0B9HLdvNyrs','0lB3NT9u8E8','0tfS7yv1Y1w','0tnG510IgP8','1-BZJ37nj7o','1d7-RT71f2g','1DYTc-Utc34','1EFucR4sG6w','1hpVmQWpI4U','1PrA4WAP9LA','1R8L-9NBO7A','1VcgR6cGmAM','2BK90Hnelp4','2GjbRr6o4ig','2hqohKDTmwQ','2NDT78IREv8','2wtdi3fas-Y','3aNUNlzLUJc','3eLKjkUoFQA','3i2uC3jRd9c','3k3a1C8d0Bo','3lNa-k_xd4s','3R5-dQ00pI4','3X0vWO3J7to','03xZWpOmDBc','3Yi-VyTdGOE','4LHYOXM8d1g','4mW0ufJQXsc','4OKZwH58Lf0','4SWZG0BGFrA','4u6O8ZktYA4','4VkmZH6tLpc','4xZvlYuVB6I','4zfvZYDfRGk','4ZjCl3ZtH40','5-v3sOxugdU','5DdE0rI2bvg','5GQLnFEt6jA','5HB_BVLxlds','5js-W-dF_PI','5pkHazhJz-s','5pWeQFLgI1k','6GeJFAMLID0','6h-QsKTgGzY','6Mjq2xilWtw','7c6X3bHOOUY','7FttAcsKYxc','7kYf4fVCWVs','7sHg5VBgnaM','7xSsGnz5V5s','8bdwKUZarrE','8BsGG4xjseo','8C4YbuYg4bU','8fA6nRprnU4','8fu5E1eGRVY','8GjhotovmFk','8GuSeiKwse0','8nx5wX2oSSI','8SZJqygjp7E','8U1ydFCBuLY','8uK_3HBsiok','8yZNuKmY9oU','9EogBcA0Jh4','9GB8hUJeOmo','9UsAmiGvxbA','17W3uOM9ijo','35x83Ilepgw','45dZMyQZOgM','57y8VR4Im54','88LQZkMuDyk','091ARUkoIeY','91MdalDprkI','A1iA8YxJJ7I','A4UEZhOQ9-s','A5cmf9KMaVs','A5Iz7vVtAGE','AcwDJEG_qJE','adA8CBDyPcc','ADsjvax9biY','AE4pGm-ov60','AIJFf_j4-G0','aKAP44ReGGo','AME90uxplKY','aNXo1H9qPHw','AO-SLA_l92Y','atsx6NC1dNI','aV5Y2lAq8Wk','aZ9ml9_ZUQ8','B1hYTluCsYQ','bbIqOtveUVM','BCEOp94te1A','BcnfMk8Sesg','BfggDpdqbhY','BgnbgC3YQQg','BLTMOHC91to','bos50SZqiLk','bp8q_W5mj9M','bPiJwDZ_syQ','BQnO07V-rdY','BSTxQ9SJBWo','bzqz-NpJbZE','c2sLOENsL0Y','C7IdbsdwQZM','C9G0HmW4ZFk','CCAKDYiQScM','cDFugg0os0A','cDmdAG_ix-o','cJG-YY755Sk','cK5nV3As-vs','CK36n0ErJAc','cL4gzPwbj5Q','cl52Rkgq3LE','cNNhb2sBEGI','crbf2MSyKF8','cw-OplECn6c','Cw9oceXD3RI','CxocFFZG9Jg','cZwFwz8Vsg0','d1VHNb2DBq4','D7LzyImMnHU','d9Jg1o-7Dk4','dhaL5UJtkfM','DhuPSMOXnI0','dizHBzXSxBQ','dME6fOlkah4','DNxiqXM5XVM','drrGSUaFO8U','dT3Xe0iXIZs','dTGtmAx0ClI','DXgPZhhKbVc','DYvhzD-u2z0','e-V7UpWnEMA','efui6me9_R0','egDG4k3uzIw','EI-5ZKNkQks','eofJ_i-J9ss','EPWiGzVrUfo','F_Thn42ORno','f0rl-wKFy-E','F2cGFCHaaR0','fat6Vo_KAsA','fEVCpCLcvdQ','FEXh9qYatOw','Ffb6XgtT-C8','fFXOccQHjLs','FGUzIhzyHmg','fmkqJl7q9L8','fmMeAmNsTjg','FNmuV--khrs','Fs_4W4wtVOw','FtP1YhNmVi4','g_wVMfoTlmI','G-LlPke9C7Y','G3inlOtC4Dc','g7DTvD99oYk','GdiYMxYR8hI','gdPTK5ndKkE','GltAhjeVedM','GnQIVDSpPXo','Gpbd3JSiJTU','gpQDaYEmvCU','gsFi1KOlLlY','gybn-I0lcsU','GZHqQW_1Ixk','h2x8vmGaH0U','HA_BulXBHBc','hdbY1xtfb5E','hIw9Zhnyk3c','HOrKVVCyNyY','hPlxd4ds_TM','hr1UgdUkp30','HrlLjx8QWHs','Hs1sFmejURY','htR1z-v_ONM','Hwmk046O0DE','hym3hdoiVHo','I-DOYf6ItJU','i0H9TmuWG6E','I5zBA8ZUCS8','I7DXbqHI_Wo','IaoP7v0ku3Q','Idt1bm5Msgg','iM701KaZ7dw','IQ-KVr-LPUY','IqR1JMAs41A','IWVjagZ2DuI','IYy4DhUxfZM','J6in4j1318w','j9uzrKduG48','JabvqN3gqSQ','JaVv2QBULFg','JCFaKgRvxNg','jDjGIcA64NY','JEFrk496o3E','JEy6kwYGPhQ','JhgzIsAyVpM','jieDRDfh1jk','jKc_i14CxZo','JnDN3eOyFWo','jnGVl2VaG-A','jonaW60lQqs','JRhb-vLmv6A','jrl-uWSopzk','jSSAX2WQieg','jt_3019YgYE','JxbbxnbPVBM','JYcwvQS5-YA','jyM1wxxrs0c','kARs_N2UteA','KayrENeGIR8','KI06_x3eVfk','KIKpjJt0u-A','kLzANpNv9W0','kOSQ80lUWvU','kZj2aEF3ATg','L-zayzyI5Pg','L6IVqfhVhOc','L6Roojw4yuQ','l9nEOtIUOkc','lE3A1ws9nuY','lg9d-BJACWA','lGRSRIiRNGw','LJ_JCI4wdSo','LjTQDFuXloA','LK-ijBpGaMY','lLS1rHerV1o','Lox2Hg7PzMc','LrXxshZ_ubs','lS_lF8qXP8A','lstC6JF9ihg','lTAKbh75nlg','lY3dc15NTww','M6ftbkDlIx0','m6MeRNpzRzo','m8x_6vggezs','M9aYpQn107o','mA4mDgw9gZs','MAY84Fy_Y-4','McGmtZHrVNw','MCvSwTODfX0','MeRa3ZMMHt8','mFYWR8e-UTo','MGOxRGD8jVI','miCGmKSnnZI','mQ8XkgZf834','MsFGOJbcTmM','mucee2MC0hU','MvNpKgXFu8s','mXdbf19Fl4o','n-IrMQJpAQk','N2CIV-qQgpk','N5eXFVbp9H8','nbQKGgEAQY4','nFxOWWj3hpM','nMGjSRtwZtQ','npJ6RV_-i7M','nqmlcQBcJmk','NrAiBsq6g_4','nSP8I5NOeLc','nWJhJVWJwZw','NwUu7v1SM54','Nx8BJbxVQ6U','nz2QwByZsD0','o-2jVOq8u4w','o7hZl8zEZNw','ObKIJgHpIRQ','OgE4Z_io4cw','okgqYSX2Dbc','OlYVrTU7rQY','oOfLaD7LyJY','oSocsFYoaGw','oTZP8Y471Wo','OwTu2aW-PPU','Oy053H3IliE','P2WZ8ZDPHdE','P5Lt6HcdApc','P6ZU65gzhVY','p8izkBjdVoY','pAWS9VO3Rls','pb2GWcSSsM8','PgnLy0VH65U','PNx8Z6NqGC0','ppbgRZ-h7l4','pPdrHhX9tHI','PrWvnNx9R4c','pTp2HhmoVok','pUL-7cTHMFs','pwzUP7LQxW4','pyCA8Iw329o','pyS0VpmhKwc','pZLDNcquUDo','q1xisuEn22E','Q5ISOJGJPiE','q5pWygulIR8','q6q0GUd3Kaw','q8Jh0E8-d98','qANJJpAb9FA','qCeyvytfi2I','QCLqY9oVwsA','qfYQgHMSouw','QmBN1Vf4CYI','qpdhy91xsKA','Qs5dLzUh5Tg','QVxkL1d0C44','QWYDT575KXM','qWZ1RJ-JL-U','r_lCn8ztUts','R3YmOAQJBwQ','r49bgPdXwf0','RcnOXaT9UXs','rf2IKNgXq-0','RgPP-9ygSVY','RJCwZGI5dt0','rL-mX8t1ZkU','RmajK5jDafw','Rn067x2Heqw','RnM8x8B9OFs','rPv91h5hzXk','rREclJ_MDKc','RtU-eN3hsLs','RTZIURVkV0c','rwErYMsQVRY','RwJCvx8xPa0','RZhVlg5AfTM','RzRHil0L_CY','S_O6j2r0UWA','S2vNelW36WE','s03oy_GnBv4','S6As0m1fb8A','Sbg1ZUTkLwo','scE1A3ttqUE','SfuIlhN_a5Y','si5bYdE-I0U','SIhI8-qrZ2Y','SiQ48gyLIFk','SKJgHCUiruo','slEuthLRYGg','smnap9ahTE8','snHxWQdNPow','SNWrQQSwkZ8','spZTz0NVbuo','srze57JDmq8','Sz81iVpXBr0','sz87UEbUdmg','t1b9KwxbgJw','TCmIcYoUZOg','tCN6MsTcfL0','tgPdcTgsUDQ','tHN4LYq_ap4','THwj_IT_Xn0','TKiiDqUnh9s','toCy_GRLM1U','ToJ7Ix2JCbg','ToKjDkaM9Qg','tqUlJ-Npz-I','tRHocBhcW8s','TUloSjh_Kgc','txdafhb-I_Y','U5EMgsBahro','Ua5yE9BuIgA','uaR2M_RIVxM','UcmYP08le_0','UCVWTlrFKHc','uGlm4lVoDqo','ui3kNuvzzPc','uKrzjv4ZeBo','UmLS7GYuKXA','UnePj-AjZaQ','UpKv2_EnpAw','uUgF3uUBR4Y','uupOgSDgI4U','UWBdfje7-nY','uWOEkMaf6C0','UwzciIEy7qk','uzJbKE1ejiw','v6G3ccw7eVk','vBk0-GNMrJU','VecGMRYsJ1A','ViddUO3M76A','ViIsrxMzaUY','VJWy2rPe_Es','VmSuVUtiyQU','vMwA0_KpIEM','VnS9zDqY-k0','VrG5ghdF5ME','W2NGM1m7Csg','W2PfEDE854E','wdYRkYf8wMs','wh_1iKMHHIs','WK503sEAxT8','Wla1VSIQ5RQ','wm1YvKCvEiE','WnUsIGL9QOc','wtA90nH3KYQ','WthbOovJxAI','WxDu1NlQE3c','X2hzTUYZEPE','x6JymSYqKzE','X8LNO0GuSlw','x9sqzVhGxaU','xBYOH9IXVwg','xdDd_PgkGiQ','xe1SWCwkPEI','xFMlOPGcpTI','xgm5yKbYDlc','XH9GtAewl6s','xkW4A3AfReQ','XM0OY6BGB8s','XMU-TTVqab8','xMVDlF6QhT0','XNztti_8iB4','xoh1X1tpNRo','XPYh7aQqmKw','xTolEZvoTNI','xuAknXYm-J8','xuapPTkaICg','XXdGhy-iNXc','Xy_tBBaNfv8','XzPvou8vq_A','y8SLGcKrNkE','Y9v8wmJK5lM','YBL3hEUmp1c','yC04Ol8NxzM','yjRF5VIRnT4','yKBTsNTljFg','Yo62nW6Q8k4','yP9jj3AheV8','YT86fF9Qa54','yW21wq5SPSw','YX-GLFfAnlM','z_sz9WpCaJs','Z-zp46uua1Y','z3N7Sjglz14','Z8jBxLkFgTA','Z80yVQmgm3A','ZcM8LrxeoOg','ZE1zI0eRFy4','zep7RdAsZrE','Zg4zmn8a-3s','ZHCagFNIEPU','ziVbfl0scXE','ZmqgtrBcQbI','zNd6OOCqwbs','ZOmZzm7YM18','zP8Oh4NLFwM','ZQj5Rio2hl8','zrh0FLVgHSY','ZtVs2UAQRJ0','ZUbzSd835CA','ZUp1Ik3acT0','zw3xSbFHwzM','zZUmVwiGvdA']
seattle_long = ['_x4YRd98rp8','-J3jTqY08Ec','-n4_aEmpInA','-OfSZdqtFnk','-ruy-w0bxvA','0DHCja5e6R8','0lB3NT9u8E8','2p0cCGoDkpI','3ldlJwuvo-4','5V0aWp2Ox0E','6KtfQQp-EHM','6O4EkKGQPjg','6WV_tr7N3Z8','8ySWHDmAvh8','9FpnUpkwYlc','45q7S74eGBc','57y8VR4Im54','78r47RkfM7M','81S1njDv6Ew','A1iA8YxJJ7I','a99AFwLZ2T0','akDZcZpXfdY','akZ6XI5fhxk','aZ9ml9_ZUQ8','b0St7tzRjls','bbIqOtveUVM','BBm3P37Atgw','BXq-VYS5PwA','c8u27orJ7PQ','C9G0HmW4ZFk','cjEaWvyIEA4','CNOyctHY1iQ','CU8VP6eIzFI','CxocFFZG9Jg','D-WTMSeCTbU','EBDZcsnIERc','EFAt7KHGaig','eofJ_i-J9ss','euUuH_t57OI','EVjdE3mjfVE','f0rl-wKFy-E','F1aziUXftZM','Ffpx2xEDfJE','fkKG2hiYzpM','H5DBrzrBPN0','IBukFngjK74','IVAKYiqVP-U','jk4JRieDWFk','jLdazmOt8SM','K5ai2ICv49A','k5liW6JxYUA','K94hUTNim1s','KIKpjJt0u-A','kLzANpNv9W0','KSOIHpnnG1Q','KZxtdLr1Y6c','L8Js1B_87oU','LK-ijBpGaMY','LwHqvzUlMw4','M9aYpQn107o','mbSYJzKd5mU','mFYWR8e-UTo','MGOxRGD8jVI','nFxOWWj3hpM','ng-kukZkUJ0','nX1zvrVp8LY','ON1gZ5M4WIE','ooyVIXV4smE','orMqgbMpOx0','oRPoMnSuZy0','oSMDC7Fr9j4','Pby0fh_wHpM','pdS2bipzIUc','Pp1gWK_Fluw','pQ4sURjktiU','PRaHgzSizKE','PVpri9TKr7c','qEC5_gA3vlM','qrDh3w22KEE','Qs5dLzUh5Tg','rDMBtpqgKAA','rv1Bf9YoaNA','TeRzXzjPLu4','TFyisEwUBVk','Tx17G2s6SDY','ui3kNuvzzPc','ut2cAzRhf5k','uzJbKE1ejiw','v_2Xpdo_Gl8','V2XnRr1RwAw','vKcfmK6UMZQ','VmSuVUtiyQU','VWyZ21CcAqk','VYP6wQeEYxI','wcQX2tv8bQ8','WUGLSJa0dPM','XJwbgKh17ls','Y9v8wmJK5lM','yAcrza_h_s0','YkCXK8kgZH4','YoVsH-caPXM','ywVdWWHZvBI','z58X4iYokB0','ZcM8LrxeoOg','ZIYUwAS9al4','ZO4UZ5qV6Tc','ZohR9JZz2ks','ZUbzSd835CA','zXL19Nbn2Eo','ZyAU50SN-f4']
chicago_long = ['_-pc4Jp_ghw','_amSXW6ER2s','-MBex64U-uA','-ruy-w0bxvA','0lB3NT9u8E8','0S1BScOkeQM','1DYTc-Utc34','1PrA4WAP9LA','2boYI6YVkCs','2vTat1k_i9I','3DjH8dOHigQ','3jyBO4V555w','3ldlJwuvo-4','5buv41nOf0k','5js-W-dF_PI','5pkHazhJz-s','6KtfQQp-EHM','6O4EkKGQPjg','7B7CL-nAhag','7EGFD50qFss','7EyK3EmBQP8','7kYf4fVCWVs','7oDSusXoeko','7YkS7NDSA4I','8fu5E1eGRVY','35x83Ilepgw','57y8VR4Im54','72HGCMl-ZBM','ACfVSGtgIaQ','agbchpkOslA','Agclcc3Z1wg','AiEATwp6J7o','aqKLErKfCIU','aZ9ml9_ZUQ8','b0St7tzRjls','b46-jgnXYKg','bbIqOtveUVM','BBm3P37Atgw','bIvWae2GDvY','BLOuG2gjosQ','BOW7YjZ7vAk','bzIAincGQko','C9G0HmW4ZFk','cjEaWvyIEA4','CU8VP6eIzFI','Dal6DC4oa_c','DhuPSMOXnI0','ef9srT0dSog','eLcwWT1UlWo','eMuFYlOK8qE','eofJ_i-J9ss','f0rl-wKFy-E','F2kZMCFZKB8','f59k6WIHnFU','Ffpx2xEDfJE','fFu9usQXULA','FZ7hT8HjLM0','GZHqQW_1Ixk','HLkYaoj8FlY','HoeIYHSJdsE','I7DXbqHI_Wo','IK0CMsio3Pk','iKxbb-Lwziw','iorJREkOEKI','J5cgxbwCll8','jBNfqcawpHA','jitaT7KmRgw','JRH5tDwuMGc','jWwH0dhDGI8','KGU3v5k6j30','kLzANpNv9W0','KuZ-ZU8o8VM','L1O4Tqtl-Ts','lHjdZiSXH6U','lixBnuPY1kU','LK-ijBpGaMY','lS_lF8qXP8A','m_1tglpJa3U','M9aYpQn107o','mA4mDgw9gZs','MFY-yVIZ04s','mFYWR8e-UTo','MGOxRGD8jVI','MsFGOJbcTmM','mzoXxaWTh9g','N5N4bHHsRKY','n7A1sSY9LQs','nFxOWWj3hpM','ng-kukZkUJ0','nHtj_mgLmEU','NsPwhY_E5pY','nUaySHRNkro','OaNTgcltRxs','oGhqerpHJ8o','Ol_-UeJqYvg','orMqgbMpOx0','oRPoMnSuZy0','p8izkBjdVoY','PSJTp8Oxygs','pSUXldFHq00','pTp2HhmoVok','pw9PaCOn13w','pwzUP7LQxW4','q8BI1I1ZFLM','Qs5dLzUh5Tg','qWZ77apWw-A','QyCJFgSTcb8','RHFqxFgy3_M','rS-xnpSlPN8','sGOaJfZ7ON0','sJ3EX6FszGE','sms_f_PYumI','snHxWQdNPow','SNWrQQSwkZ8','t1b9KwxbgJw','T8aJ5w_y6Wg','tfHqbXkJlRM','tHN4LYq_ap4','THsmZ_kXU4s','TX_a8qmqMPY','u9zGFpGeJY0','u11hu5YeiSI','U99cl9oZ9OM','Ua5yE9BuIgA','uGlm4lVoDqo','UkWYNbstUxM','UUGdULch9J8','uWOEkMaf6C0','uZaaquVFX3o','uzJbKE1ejiw','V2XnRr1RwAw','V7q_sqL1Bio','v8bmJbsgUXo','v19JaO5RPVw','VCgiuweaqS0','VfP4kzjEKoE','VPkHGd6K2pQ','VWok20BmvZc','W2NGM1m7Csg','Wg6Gwrr6bnI','wpNa8R93FJs','ww6Hv-XlkAE','wZfMDlW99Us','Xd9hGuHQi7I','XH9GtAewl6s','XNd_CpZX1xc','y-79SBEY9vA','Y9v8wmJK5lM','yANb4fFskQU','YIzu8LzuZSY','YL46jPPx84k','YrRblZnnP9c','Yz5ZizKgIKw','z7b8R9lgq8g','zNd6OOCqwbs','ZUbzSd835CA','ZUFqN8rPpD8']
all_videos_unique_long = list(set(ny_long+seattle_long+chicago_long))

filenames = seattle_long

subtitle_dic = {}

def read_csv(csv_name):
	with open(csv_name, newline='', encoding="utf8") as f:
		reader = csv.reader(f)
		tmp = list(reader)
	return tmp

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
def lemmatization(doc, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
	texts_out = []
	doc = nlp(doc)
	texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
	return texts_out

for root, dirs, files in os.walk('../Transcripts'):
	for file in files:
		if file.endswith('.txt') and (file.split('.')[0] in filenames):
			with open(os.path.join(root, file)) as f:
				lines = f.readlines()
			script = lines[0]
			stop_words = set(stopwords.words('english'))
			word_tokens = word_tokenize(script)
			script = [w for w in word_tokens if not w in stop_words]
			script = TreebankWordDetokenizer().detokenize(script)
			script = list(gensim.utils.simple_preprocess(str(script), deacc=False))
			script = " ".join(script)
			script = lemmatization(script)
			subtitle_dic[file.split('.')[0]] = script[0]


# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, verbose=True, n_words=20):
	# vectorizer, lda_model = best_LDA.get_model()
	keywords = np.array(vectorizer.get_feature_names())
	topic_keywords = []
	for topic_weights in lda_model.components_:
		top_keyword_locs = (-topic_weights).argsort()[:n_words]
		topic_keywords.append(keywords.take(top_keyword_locs))
	if verbose:
		for i in range(0, len(topic_keywords)):
			print("Topic " + str(i))
			print(list(topic_keywords[i]))
	return topic_keywords



train_data = []; collcted_video_id = []
for video_id in filenames:
	try:
		train_data.append(subtitle_dic[video_id])
		collcted_video_id.append(video_id)
	except:
		pass;

score_all = []; time_all = []

class LDA_classifier(BaseEstimator, ClassifierMixin):

	def __init__(self, max_df, min_df, topic_number, learning_decay, learning_offset, batch_size, max_iter, score_type):
	# def __init__(self):
		self.max_df = max_df
		self.min_df = min_df
		self.topic_number = topic_number
		self.learning_decay = learning_decay
		self.learning_offset = learning_offset
		self.batch_size = batch_size
		self.max_iter = max_iter
		self.score_type = score_type
		self.embedding_model = embedding_model
		
	def fit(self, train_data):
		print('fitting:', self.max_df, self.min_df, self.topic_number, self.learning_decay, self.learning_offset, self.batch_size, self.max_iter)
		self.vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df)
		self.lda_model = LatentDirichletAllocation(n_components=self.topic_number, learning_decay = self.learning_decay,
                                                	learning_offset = self.learning_offset, batch_size = self.batch_size,           
													learning_method='online', random_state=100, max_iter=self.max_iter)
		data_vectorized = self.vectorizer.fit_transform(train_data)
		self.lda_model.fit(data_vectorized)

	def predict(self, texts):
		text_vectorized = self.vectorizer.transform(texts)
		return self.lda_model.transform(text_vectorized)

	def score(self, train_data):
		if (self.score_type == 'perplexity'):
			tmp = self.vectorizer.transform(train_data)
			score = -self.lda_model.perplexity(tmp)
			print('scoring:', score)
			# global score_all, time_all
			time_all.append(time.time() - time_all[0])
			score_all.append(max([score]+score_all))
		elif (self.score_type == 'coherence'):
			score = metric_coherence_gensim(measure='u_mass', topic_word_distrib=self.lda_model.components_, 
				vocab=np.array(self.vectorizer.get_feature_names()), dtm=self.vectorizer.transform(train_data), return_mean=True)
			print(score)
		elif (self.score_type == 'embedding'):
			topic_keywords = show_topics(self.vectorizer, self.lda_model, verbose=False, n_words=10)
			keywords = []
			for i in range(0, len(topic_keywords)):
				keywords.append(list(topic_keywords[i]))
			score = embedding_distance(keywords, self.embedding_model)
			print('scoring:', score)
		return score

	def get_model(self):
		# print('the final model:', self.max_df, self.min_df, self.topic_number)
		return (self.vectorizer, self.lda_model)

if (not load_model):
	cv = [(slice(None), slice(None))]
	lda = LDA_classifier(max_df=0.1, min_df=0.05, topic_number=5, learning_decay=0.1, learning_offset=0.1, batch_size=64, max_iter=100, score_type=score_type, embedding_model=embedding_model)
	grid_search = GridSearchCV(lda, param_grid=param_dist, cv=cv, n_jobs=1)

	time_all.append(time.time())
	grid_search.fit(train_data)
	print(grid_search.best_params_)

	time_all = time_all[1:]
	plt.plot(time_all, score_all)
	plt.xlabel('time')
	plt.ylabel('score')
	# plt.show()
	# best_LDA.get_model()
	# best_LDA.score(train_data)

	if save_to_model:
		best_LDA = grid_search.best_estimator_
		with open('best_LDA.pkl', 'wb') as f:
			pickle.dump(best_LDA, f)

else:
	with open('best_LDA.pkl', 'rb') as f:
		best_LDA = pickle.load(f)
	best_LDA.score(train_data)

# vectorizer, lda_model = best_LDA.get_model()
# show_topics(vectorizer, lda_model)





