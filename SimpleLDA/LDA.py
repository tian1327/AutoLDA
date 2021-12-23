import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
import gensim
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
import time
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import sys
import csv
import matplotlib.pyplot as plt

np.set_printoptions(threshold= sys.maxsize)
np.random.seed(5)

load_model = True
model_name = 'lda_model.pkl'
save_to_model = False
save_model_name = 'lda_model.pkl'

topic_number = 6

ny = ['UnePj-AjZaQ','5pkHazhJz-s','jnGVl2VaG-A','dhaL5UJtkfM','eofJ_i-J9ss','dME6fOlkah4','PrWvnNx9R4c','KIKpjJt0u-A','__aNiRlpOtM','4u6O8ZktYA4','0B9HLdvNyrs','TKiiDqUnh9s','NwUu7v1SM54','57y8VR4Im54','ViddUO3M76A','8yZNuKmY9oU','GnQIVDSpPXo','dizHBzXSxBQ','ZmqgtrBcQbI','xBYOH9IXVwg','xFMlOPGcpTI','Lox2Hg7PzMc','FtP1YhNmVi4','LK-ijBpGaMY','A1iA8YxJJ7I','2GjbRr6o4ig','3lNa-k_xd4s','fat6Vo_KAsA','F2cGFCHaaR0','lY3dc15NTww','nWJhJVWJwZw','0lB3NT9u8E8','jSSAX2WQieg','3X0vWO3J7to','KayrENeGIR8','PgnLy0VH65U','BQnO07V-rdY','SKJgHCUiruo','4mW0ufJQXsc','qfYQgHMSouw','2NDT78IREv8','kZj2aEF3ATg','L-zayzyI5Pg','hr1UgdUkp30','cZwFwz8Vsg0','G-LlPke9C7Y','I5zBA8ZUCS8','UWBdfje7-nY','1R8L-9NBO7A','oOfLaD7LyJY','uUgF3uUBR4Y','JEy6kwYGPhQ','oSocsFYoaGw','pb2GWcSSsM8','W2PfEDE854E','CCAKDYiQScM','ToKjDkaM9Qg','OwTu2aW-PPU','l9nEOtIUOkc','5HB_BVLxlds','RcnOXaT9UXs','GdiYMxYR8hI','9EogBcA0Jh4','QVxkL1d0C44','okgqYSX2Dbc','v6G3ccw7eVk','P5Lt6HcdApc','lTAKbh75nlg','7xSsGnz5V5s','dT3Xe0iXIZs','nqmlcQBcJmk','txdafhb-I_Y','tCN6MsTcfL0','tqUlJ-Npz-I','lGRSRIiRNGw','yjRF5VIRnT4','JhgzIsAyVpM','yKBTsNTljFg','1hpVmQWpI4U','JRhb-vLmv6A','R3YmOAQJBwQ','_2C4tWdpdfE','zP8Oh4NLFwM','4LHYOXM8d1g','uupOgSDgI4U','JEFrk496o3E','U5EMgsBahro','B1hYTluCsYQ','o7hZl8zEZNw','gybn-I0lcsU','AcwDJEG_qJE','1d7-RT71f2g','TCmIcYoUZOg','BCEOp94te1A','zrh0FLVgHSY','W2NGM1m7Csg','YBL3hEUmp1c','gsFi1KOlLlY','RwJCvx8xPa0','ZE1zI0eRFy4','ZtVs2UAQRJ0','h2x8vmGaH0U','adA8CBDyPcc','8bdwKUZarrE','4VkmZH6tLpc','oTZP8Y471Wo','kLzANpNv9W0','pZLDNcquUDo','QWYDT575KXM','uGlm4lVoDqo','7kYf4fVCWVs','Sbg1ZUTkLwo','IqR1JMAs41A','xuapPTkaICg','uKrzjv4ZeBo','yC04Ol8NxzM','VnS9zDqY-k0','5js-W-dF_PI','j9uzrKduG48','8fu5E1eGRVY','ADsjvax9biY','1PrA4WAP9LA','VrG5ghdF5ME','jonaW60lQqs','2BK90Hnelp4','jKc_i14CxZo','XNztti_8iB4','qWZ1RJ-JL-U','OgE4Z_io4cw','HrlLjx8QWHs','XMU-TTVqab8','OlYVrTU7rQY','1EFucR4sG6w','Z-zp46uua1Y','JabvqN3gqSQ','crbf2MSyKF8','HOrKVVCyNyY','pPdrHhX9tHI','lLS1rHerV1o','AE4pGm-ov60','si5bYdE-I0U','ToJ7Ix2JCbg','lstC6JF9ihg','M6ftbkDlIx0','Fs_4W4wtVOw','zep7RdAsZrE','5DdE0rI2bvg','9GB8hUJeOmo','cNNhb2sBEGI','vBk0-GNMrJU','ViIsrxMzaUY','xe1SWCwkPEI','BLTMOHC91to','lg9d-BJACWA','Q5ISOJGJPiE','WnUsIGL9QOc','IWVjagZ2DuI','pyS0VpmhKwc','smnap9ahTE8','3i2uC3jRd9c','McGmtZHrVNw','ui3kNuvzzPc','8BsGG4xjseo','wtA90nH3KYQ','_QJQzD6Qgkk','UcmYP08le_0','RTZIURVkV0c','egDG4k3uzIw','q8Jh0E8-d98','q1xisuEn22E','yP9jj3AheV8','MCvSwTODfX0','g_wVMfoTlmI','Ffb6XgtT-C8','RnM8x8B9OFs','1VcgR6cGmAM','03xZWpOmDBc','ZUp1Ik3acT0','KI06_x3eVfk','FNmuV--khrs','FEXh9qYatOw','EI-5ZKNkQks','r_lCn8ztUts','miCGmKSnnZI','Xy_tBBaNfv8','JxbbxnbPVBM','dTGtmAx0ClI','RgPP-9ygSVY','45dZMyQZOgM','sz87UEbUdmg','ziVbfl0scXE','rREclJ_MDKc','aNXo1H9qPHw','fmMeAmNsTjg','5pWeQFLgI1k','QmBN1Vf4CYI','xTolEZvoTNI','tRHocBhcW8s','SfuIlhN_a5Y','C7IdbsdwQZM','aV5Y2lAq8Wk','L6Roojw4yuQ','r49bgPdXwf0','nFxOWWj3hpM','pUL-7cTHMFs','UwzciIEy7qk','8GuSeiKwse0','X2hzTUYZEPE','drrGSUaFO8U','UpKv2_EnpAw','7sHg5VBgnaM','TUloSjh_Kgc','YX-GLFfAnlM','bp8q_W5mj9M','kARs_N2UteA','_Nst1BllAx8','n-IrMQJpAQk','J6in4j1318w','IQ-KVr-LPUY','WthbOovJxAI','091ARUkoIeY','LjTQDFuXloA','rf2IKNgXq-0','Rn067x2Heqw','xgm5yKbYDlc','mA4mDgw9gZs','9UsAmiGvxbA','z_sz9WpCaJs','7c6X3bHOOUY','S_O6j2r0UWA','xuAknXYm-J8','bPiJwDZ_syQ','_GLjAs7rvNo','I-DOYf6ItJU','p8izkBjdVoY','5GQLnFEt6jA','AO-SLA_l92Y','wdYRkYf8wMs','DYvhzD-u2z0','RzRHil0L_CY','RmajK5jDafw','PNx8Z6NqGC0','pAWS9VO3Rls','EPWiGzVrUfo','scE1A3ttqUE','htR1z-v_ONM','A4UEZhOQ9-s']

seattle = ['pQ4sURjktiU','pdS2bipzIUc','ZcM8LrxeoOg','-J3jTqY08Ec','akZ6XI5fhxk','9FpnUpkwYlc','K5ai2ICv49A','YkCXK8kgZH4','PVpri9TKr7c','mbSYJzKd5mU','vKcfmK6UMZQ','KSOIHpnnG1Q','ZyAU50SN-f4','-n4_aEmpInA','LwHqvzUlMw4','PRaHgzSizKE','z58X4iYokB0','ywVdWWHZvBI','D-WTMSeCTbU','Tx17G2s6SDY','wcQX2tv8bQ8','zXL19Nbn2Eo','ooyVIXV4smE','jk4JRieDWFk','nX1zvrVp8LY','K94hUTNim1s','Pby0fh_wHpM','VWyZ21CcAqk','c8u27orJ7PQ','TFyisEwUBVk','euUuH_t57OI','L8Js1B_87oU','CxocFFZG9Jg','81S1njDv6Ew','F1aziUXftZM','rv1Bf9YoaNA','jLdazmOt8SM','a99AFwLZ2T0','XJwbgKh17ls','45q7S74eGBc','6WV_tr7N3Z8','EBDZcsnIERc','ON1gZ5M4WIE','CNOyctHY1iQ','qrDh3w22KEE','0DHCja5e6R8','VYP6wQeEYxI','ZIYUwAS9al4','ut2cAzRhf5k','KZxtdLr1Y6c','ZO4UZ5qV6Tc','rDMBtpqgKAA','Pp1gWK_Fluw','WUGLSJa0dPM','akDZcZpXfdY','5V0aWp2Ox0E','YoVsH-caPXM','2p0cCGoDkpI','EFAt7KHGaig','IBukFngjK74','-OfSZdqtFnk','H5DBrzrBPN0','ZohR9JZz2ks','_x4YRd98rp8','IVAKYiqVP-U','qEC5_gA3vlM','v_2Xpdo_Gl8','8ySWHDmAvh8','oSMDC7Fr9j4','BXq-VYS5PwA','TeRzXzjPLu4']

chicago = ['KuZ-ZU8o8VM','BLOuG2gjosQ','DhuPSMOXnI0','PSJTp8Oxygs','ZUFqN8rPpD8','MsFGOJbcTmM','ACfVSGtgIaQ','VCgiuweaqS0','7EyK3EmBQP8','uWOEkMaf6C0','pTp2HhmoVok','ef9srT0dSog','NsPwhY_E5pY','qWZ77apWw-A','wpNa8R93FJs','0S1BScOkeQM','3jyBO4V555w','MFY-yVIZ04s','THsmZ_kXU4s','HoeIYHSJdsE','1DYTc-Utc34','sGOaJfZ7ON0','7YkS7NDSA4I','Ol_-UeJqYvg','_amSXW6ER2s','RHFqxFgy3_M','t1b9KwxbgJw','L1O4Tqtl-Ts','nHtj_mgLmEU','UUGdULch9J8','J5cgxbwCll8','UkWYNbstUxM','VPkHGd6K2pQ','3DjH8dOHigQ','KGU3v5k6j30','agbchpkOslA','F2kZMCFZKB8','TX_a8qmqMPY','iKxbb-Lwziw','VWok20BmvZc','v19JaO5RPVw','2vTat1k_i9I','QyCJFgSTcb8','2boYI6YVkCs','sms_f_PYumI','U99cl9oZ9OM','Yz5ZizKgIKw','35x83Ilepgw','HLkYaoj8FlY','tfHqbXkJlRM','IK0CMsio3Pk','VfP4kzjEKoE','mzoXxaWTh9g','n7A1sSY9LQs','pwzUP7LQxW4','m_1tglpJa3U','fFu9usQXULA','sJ3EX6FszGE','tHN4LYq_ap4','Xd9hGuHQi7I','zNd6OOCqwbs','eMuFYlOK8qE','XNd_CpZX1xc','nUaySHRNkro','yANb4fFskQU','YL46jPPx84k','y-79SBEY9vA','bIvWae2GDvY','YrRblZnnP9c','T8aJ5w_y6Wg','aqKLErKfCIU','b46-jgnXYKg','Ua5yE9BuIgA','lHjdZiSXH6U','z7b8R9lgq8g','snHxWQdNPow','ww6Hv-XlkAE','v8bmJbsgUXo','u9zGFpGeJY0','-MBex64U-uA','bzIAincGQko','pw9PaCOn13w','N5N4bHHsRKY','JRH5tDwuMGc','Wg6Gwrr6bnI','pSUXldFHq00','I7DXbqHI_Wo','Dal6DC4oa_c','oGhqerpHJ8o','q8BI1I1ZFLM','7B7CL-nAhag','Agclcc3Z1wg','YIzu8LzuZSY','u11hu5YeiSI','uZaaquVFX3o','7EGFD50qFss','BOW7YjZ7vAk','FZ7hT8HjLM0','jitaT7KmRgw','XH9GtAewl6s','rS-xnpSlPN8','7oDSusXoeko','f59k6WIHnFU','_-pc4Jp_ghw','lixBnuPY1kU','jBNfqcawpHA','wZfMDlW99Us','72HGCMl-ZBM','AiEATwp6J7o','GZHqQW_1Ixk','b0St7tzRjls']

ny_long = ['__aNiRlpOtM','_2C4tWdpdfE','_5Dlspmarck','_dlVkCeP3y4','_DNudv1Rtfg','_GLjAs7rvNo','_mOhVtoBUUg','_Nst1BllAx8','_QJQzD6Qgkk','-1H2myJkHXY','-e0j4Oz_kOg','-klLyOizv8c','-ruy-w0bxvA','-RWCteAeCsU','0B9HLdvNyrs','0lB3NT9u8E8','0tfS7yv1Y1w','0tnG510IgP8','1-BZJ37nj7o','1d7-RT71f2g','1DYTc-Utc34','1EFucR4sG6w','1hpVmQWpI4U','1PrA4WAP9LA','1R8L-9NBO7A','1VcgR6cGmAM','2BK90Hnelp4','2GjbRr6o4ig','2hqohKDTmwQ','2NDT78IREv8','2wtdi3fas-Y','3aNUNlzLUJc','3eLKjkUoFQA','3i2uC3jRd9c','3k3a1C8d0Bo','3lNa-k_xd4s','3R5-dQ00pI4','3X0vWO3J7to','03xZWpOmDBc','3Yi-VyTdGOE','4LHYOXM8d1g','4mW0ufJQXsc','4OKZwH58Lf0','4SWZG0BGFrA','4u6O8ZktYA4','4VkmZH6tLpc','4xZvlYuVB6I','4zfvZYDfRGk','4ZjCl3ZtH40','5-v3sOxugdU','5DdE0rI2bvg','5GQLnFEt6jA','5HB_BVLxlds','5js-W-dF_PI','5pkHazhJz-s','5pWeQFLgI1k','6GeJFAMLID0','6h-QsKTgGzY','6Mjq2xilWtw','7c6X3bHOOUY','7FttAcsKYxc','7kYf4fVCWVs','7sHg5VBgnaM','7xSsGnz5V5s','8bdwKUZarrE','8BsGG4xjseo','8C4YbuYg4bU','8fA6nRprnU4','8fu5E1eGRVY','8GjhotovmFk','8GuSeiKwse0','8nx5wX2oSSI','8SZJqygjp7E','8U1ydFCBuLY','8uK_3HBsiok','8yZNuKmY9oU','9EogBcA0Jh4','9GB8hUJeOmo','9UsAmiGvxbA','17W3uOM9ijo','35x83Ilepgw','45dZMyQZOgM','57y8VR4Im54','88LQZkMuDyk','091ARUkoIeY','91MdalDprkI','A1iA8YxJJ7I','A4UEZhOQ9-s','A5cmf9KMaVs','A5Iz7vVtAGE','AcwDJEG_qJE','adA8CBDyPcc','ADsjvax9biY','AE4pGm-ov60','AIJFf_j4-G0','aKAP44ReGGo','AME90uxplKY','aNXo1H9qPHw','AO-SLA_l92Y','atsx6NC1dNI','aV5Y2lAq8Wk','aZ9ml9_ZUQ8','B1hYTluCsYQ','bbIqOtveUVM','BCEOp94te1A','BcnfMk8Sesg','BfggDpdqbhY','BgnbgC3YQQg','BLTMOHC91to','bos50SZqiLk','bp8q_W5mj9M','bPiJwDZ_syQ','BQnO07V-rdY','BSTxQ9SJBWo','bzqz-NpJbZE','c2sLOENsL0Y','C7IdbsdwQZM','C9G0HmW4ZFk','CCAKDYiQScM','cDFugg0os0A','cDmdAG_ix-o','cJG-YY755Sk','cK5nV3As-vs','CK36n0ErJAc','cL4gzPwbj5Q','cl52Rkgq3LE','cNNhb2sBEGI','crbf2MSyKF8','cw-OplECn6c','Cw9oceXD3RI','CxocFFZG9Jg','cZwFwz8Vsg0','d1VHNb2DBq4','D7LzyImMnHU','d9Jg1o-7Dk4','dhaL5UJtkfM','DhuPSMOXnI0','dizHBzXSxBQ','dME6fOlkah4','DNxiqXM5XVM','drrGSUaFO8U','dT3Xe0iXIZs','dTGtmAx0ClI','DXgPZhhKbVc','DYvhzD-u2z0','e-V7UpWnEMA','efui6me9_R0','egDG4k3uzIw','EI-5ZKNkQks','eofJ_i-J9ss','EPWiGzVrUfo','F_Thn42ORno','f0rl-wKFy-E','F2cGFCHaaR0','fat6Vo_KAsA','fEVCpCLcvdQ','FEXh9qYatOw','Ffb6XgtT-C8','fFXOccQHjLs','FGUzIhzyHmg','fmkqJl7q9L8','fmMeAmNsTjg','FNmuV--khrs','Fs_4W4wtVOw','FtP1YhNmVi4','g_wVMfoTlmI','G-LlPke9C7Y','G3inlOtC4Dc','g7DTvD99oYk','GdiYMxYR8hI','gdPTK5ndKkE','GltAhjeVedM','GnQIVDSpPXo','Gpbd3JSiJTU','gpQDaYEmvCU','gsFi1KOlLlY','gybn-I0lcsU','GZHqQW_1Ixk','h2x8vmGaH0U','HA_BulXBHBc','hdbY1xtfb5E','hIw9Zhnyk3c','HOrKVVCyNyY','hPlxd4ds_TM','hr1UgdUkp30','HrlLjx8QWHs','Hs1sFmejURY','htR1z-v_ONM','Hwmk046O0DE','hym3hdoiVHo','I-DOYf6ItJU','i0H9TmuWG6E','I5zBA8ZUCS8','I7DXbqHI_Wo','IaoP7v0ku3Q','Idt1bm5Msgg','iM701KaZ7dw','IQ-KVr-LPUY','IqR1JMAs41A','IWVjagZ2DuI','IYy4DhUxfZM','J6in4j1318w','j9uzrKduG48','JabvqN3gqSQ','JaVv2QBULFg','JCFaKgRvxNg','jDjGIcA64NY','JEFrk496o3E','JEy6kwYGPhQ','JhgzIsAyVpM','jieDRDfh1jk','jKc_i14CxZo','JnDN3eOyFWo','jnGVl2VaG-A','jonaW60lQqs','JRhb-vLmv6A','jrl-uWSopzk','jSSAX2WQieg','jt_3019YgYE','JxbbxnbPVBM','JYcwvQS5-YA','jyM1wxxrs0c','kARs_N2UteA','KayrENeGIR8','KI06_x3eVfk','KIKpjJt0u-A','kLzANpNv9W0','kOSQ80lUWvU','kZj2aEF3ATg','L-zayzyI5Pg','L6IVqfhVhOc','L6Roojw4yuQ','l9nEOtIUOkc','lE3A1ws9nuY','lg9d-BJACWA','lGRSRIiRNGw','LJ_JCI4wdSo','LjTQDFuXloA','LK-ijBpGaMY','lLS1rHerV1o','Lox2Hg7PzMc','LrXxshZ_ubs','lS_lF8qXP8A','lstC6JF9ihg','lTAKbh75nlg','lY3dc15NTww','M6ftbkDlIx0','m6MeRNpzRzo','m8x_6vggezs','M9aYpQn107o','mA4mDgw9gZs','MAY84Fy_Y-4','McGmtZHrVNw','MCvSwTODfX0','MeRa3ZMMHt8','mFYWR8e-UTo','MGOxRGD8jVI','miCGmKSnnZI','mQ8XkgZf834','MsFGOJbcTmM','mucee2MC0hU','MvNpKgXFu8s','mXdbf19Fl4o','n-IrMQJpAQk','N2CIV-qQgpk','N5eXFVbp9H8','nbQKGgEAQY4','nFxOWWj3hpM','nMGjSRtwZtQ','npJ6RV_-i7M','nqmlcQBcJmk','NrAiBsq6g_4','nSP8I5NOeLc','nWJhJVWJwZw','NwUu7v1SM54','Nx8BJbxVQ6U','nz2QwByZsD0','o-2jVOq8u4w','o7hZl8zEZNw','ObKIJgHpIRQ','OgE4Z_io4cw','okgqYSX2Dbc','OlYVrTU7rQY','oOfLaD7LyJY','oSocsFYoaGw','oTZP8Y471Wo','OwTu2aW-PPU','Oy053H3IliE','P2WZ8ZDPHdE','P5Lt6HcdApc','P6ZU65gzhVY','p8izkBjdVoY','pAWS9VO3Rls','pb2GWcSSsM8','PgnLy0VH65U','PNx8Z6NqGC0','ppbgRZ-h7l4','pPdrHhX9tHI','PrWvnNx9R4c','pTp2HhmoVok','pUL-7cTHMFs','pwzUP7LQxW4','pyCA8Iw329o','pyS0VpmhKwc','pZLDNcquUDo','q1xisuEn22E','Q5ISOJGJPiE','q5pWygulIR8','q6q0GUd3Kaw','q8Jh0E8-d98','qANJJpAb9FA','qCeyvytfi2I','QCLqY9oVwsA','qfYQgHMSouw','QmBN1Vf4CYI','qpdhy91xsKA','Qs5dLzUh5Tg','QVxkL1d0C44','QWYDT575KXM','qWZ1RJ-JL-U','r_lCn8ztUts','R3YmOAQJBwQ','r49bgPdXwf0','RcnOXaT9UXs','rf2IKNgXq-0','RgPP-9ygSVY','RJCwZGI5dt0','rL-mX8t1ZkU','RmajK5jDafw','Rn067x2Heqw','RnM8x8B9OFs','rPv91h5hzXk','rREclJ_MDKc','RtU-eN3hsLs','RTZIURVkV0c','rwErYMsQVRY','RwJCvx8xPa0','RZhVlg5AfTM','RzRHil0L_CY','S_O6j2r0UWA','S2vNelW36WE','s03oy_GnBv4','S6As0m1fb8A','Sbg1ZUTkLwo','scE1A3ttqUE','SfuIlhN_a5Y','si5bYdE-I0U','SIhI8-qrZ2Y','SiQ48gyLIFk','SKJgHCUiruo','slEuthLRYGg','smnap9ahTE8','snHxWQdNPow','SNWrQQSwkZ8','spZTz0NVbuo','srze57JDmq8','Sz81iVpXBr0','sz87UEbUdmg','t1b9KwxbgJw','TCmIcYoUZOg','tCN6MsTcfL0','tgPdcTgsUDQ','tHN4LYq_ap4','THwj_IT_Xn0','TKiiDqUnh9s','toCy_GRLM1U','ToJ7Ix2JCbg','ToKjDkaM9Qg','tqUlJ-Npz-I','tRHocBhcW8s','TUloSjh_Kgc','txdafhb-I_Y','U5EMgsBahro','Ua5yE9BuIgA','uaR2M_RIVxM','UcmYP08le_0','UCVWTlrFKHc','uGlm4lVoDqo','ui3kNuvzzPc','uKrzjv4ZeBo','UmLS7GYuKXA','UnePj-AjZaQ','UpKv2_EnpAw','uUgF3uUBR4Y','uupOgSDgI4U','UWBdfje7-nY','uWOEkMaf6C0','UwzciIEy7qk','uzJbKE1ejiw','v6G3ccw7eVk','vBk0-GNMrJU','VecGMRYsJ1A','ViddUO3M76A','ViIsrxMzaUY','VJWy2rPe_Es','VmSuVUtiyQU','vMwA0_KpIEM','VnS9zDqY-k0','VrG5ghdF5ME','W2NGM1m7Csg','W2PfEDE854E','wdYRkYf8wMs','wh_1iKMHHIs','WK503sEAxT8','Wla1VSIQ5RQ','wm1YvKCvEiE','WnUsIGL9QOc','wtA90nH3KYQ','WthbOovJxAI','WxDu1NlQE3c','X2hzTUYZEPE','x6JymSYqKzE','X8LNO0GuSlw','x9sqzVhGxaU','xBYOH9IXVwg','xdDd_PgkGiQ','xe1SWCwkPEI','xFMlOPGcpTI','xgm5yKbYDlc','XH9GtAewl6s','xkW4A3AfReQ','XM0OY6BGB8s','XMU-TTVqab8','xMVDlF6QhT0','XNztti_8iB4','xoh1X1tpNRo','XPYh7aQqmKw','xTolEZvoTNI','xuAknXYm-J8','xuapPTkaICg','XXdGhy-iNXc','Xy_tBBaNfv8','XzPvou8vq_A','y8SLGcKrNkE','Y9v8wmJK5lM','YBL3hEUmp1c','yC04Ol8NxzM','yjRF5VIRnT4','yKBTsNTljFg','Yo62nW6Q8k4','yP9jj3AheV8','YT86fF9Qa54','yW21wq5SPSw','YX-GLFfAnlM','z_sz9WpCaJs','Z-zp46uua1Y','z3N7Sjglz14','Z8jBxLkFgTA','Z80yVQmgm3A','ZcM8LrxeoOg','ZE1zI0eRFy4','zep7RdAsZrE','Zg4zmn8a-3s','ZHCagFNIEPU','ziVbfl0scXE','ZmqgtrBcQbI','zNd6OOCqwbs','ZOmZzm7YM18','zP8Oh4NLFwM','ZQj5Rio2hl8','zrh0FLVgHSY','ZtVs2UAQRJ0','ZUbzSd835CA','ZUp1Ik3acT0','zw3xSbFHwzM','zZUmVwiGvdA']
seattle_long = ['_x4YRd98rp8','-J3jTqY08Ec','-n4_aEmpInA','-OfSZdqtFnk','-ruy-w0bxvA','0DHCja5e6R8','0lB3NT9u8E8','2p0cCGoDkpI','3ldlJwuvo-4','5V0aWp2Ox0E','6KtfQQp-EHM','6O4EkKGQPjg','6WV_tr7N3Z8','8ySWHDmAvh8','9FpnUpkwYlc','45q7S74eGBc','57y8VR4Im54','78r47RkfM7M','81S1njDv6Ew','A1iA8YxJJ7I','a99AFwLZ2T0','akDZcZpXfdY','akZ6XI5fhxk','aZ9ml9_ZUQ8','b0St7tzRjls','bbIqOtveUVM','BBm3P37Atgw','BXq-VYS5PwA','c8u27orJ7PQ','C9G0HmW4ZFk','cjEaWvyIEA4','CNOyctHY1iQ','CU8VP6eIzFI','CxocFFZG9Jg','D-WTMSeCTbU','EBDZcsnIERc','EFAt7KHGaig','eofJ_i-J9ss','euUuH_t57OI','EVjdE3mjfVE','f0rl-wKFy-E','F1aziUXftZM','Ffpx2xEDfJE','fkKG2hiYzpM','H5DBrzrBPN0','IBukFngjK74','IVAKYiqVP-U','jk4JRieDWFk','jLdazmOt8SM','K5ai2ICv49A','k5liW6JxYUA','K94hUTNim1s','KIKpjJt0u-A','kLzANpNv9W0','KSOIHpnnG1Q','KZxtdLr1Y6c','L8Js1B_87oU','LK-ijBpGaMY','LwHqvzUlMw4','M9aYpQn107o','mbSYJzKd5mU','mFYWR8e-UTo','MGOxRGD8jVI','nFxOWWj3hpM','ng-kukZkUJ0','nX1zvrVp8LY','ON1gZ5M4WIE','ooyVIXV4smE','orMqgbMpOx0','oRPoMnSuZy0','oSMDC7Fr9j4','Pby0fh_wHpM','pdS2bipzIUc','Pp1gWK_Fluw','pQ4sURjktiU','PRaHgzSizKE','PVpri9TKr7c','qEC5_gA3vlM','qrDh3w22KEE','Qs5dLzUh5Tg','rDMBtpqgKAA','rv1Bf9YoaNA','TeRzXzjPLu4','TFyisEwUBVk','Tx17G2s6SDY','ui3kNuvzzPc','ut2cAzRhf5k','uzJbKE1ejiw','v_2Xpdo_Gl8','V2XnRr1RwAw','vKcfmK6UMZQ','VmSuVUtiyQU','VWyZ21CcAqk','VYP6wQeEYxI','wcQX2tv8bQ8','WUGLSJa0dPM','XJwbgKh17ls','Y9v8wmJK5lM','yAcrza_h_s0','YkCXK8kgZH4','YoVsH-caPXM','ywVdWWHZvBI','z58X4iYokB0','ZcM8LrxeoOg','ZIYUwAS9al4','ZO4UZ5qV6Tc','ZohR9JZz2ks','ZUbzSd835CA','zXL19Nbn2Eo','ZyAU50SN-f4']
chicago_long = ['_-pc4Jp_ghw','_amSXW6ER2s','-MBex64U-uA','-ruy-w0bxvA','0lB3NT9u8E8','0S1BScOkeQM','1DYTc-Utc34','1PrA4WAP9LA','2boYI6YVkCs','2vTat1k_i9I','3DjH8dOHigQ','3jyBO4V555w','3ldlJwuvo-4','5buv41nOf0k','5js-W-dF_PI','5pkHazhJz-s','6KtfQQp-EHM','6O4EkKGQPjg','7B7CL-nAhag','7EGFD50qFss','7EyK3EmBQP8','7kYf4fVCWVs','7oDSusXoeko','7YkS7NDSA4I','8fu5E1eGRVY','35x83Ilepgw','57y8VR4Im54','72HGCMl-ZBM','ACfVSGtgIaQ','agbchpkOslA','Agclcc3Z1wg','AiEATwp6J7o','aqKLErKfCIU','aZ9ml9_ZUQ8','b0St7tzRjls','b46-jgnXYKg','bbIqOtveUVM','BBm3P37Atgw','bIvWae2GDvY','BLOuG2gjosQ','BOW7YjZ7vAk','bzIAincGQko','C9G0HmW4ZFk','cjEaWvyIEA4','CU8VP6eIzFI','Dal6DC4oa_c','DhuPSMOXnI0','ef9srT0dSog','eLcwWT1UlWo','eMuFYlOK8qE','eofJ_i-J9ss','f0rl-wKFy-E','F2kZMCFZKB8','f59k6WIHnFU','Ffpx2xEDfJE','fFu9usQXULA','FZ7hT8HjLM0','GZHqQW_1Ixk','HLkYaoj8FlY','HoeIYHSJdsE','I7DXbqHI_Wo','IK0CMsio3Pk','iKxbb-Lwziw','iorJREkOEKI','J5cgxbwCll8','jBNfqcawpHA','jitaT7KmRgw','JRH5tDwuMGc','jWwH0dhDGI8','KGU3v5k6j30','kLzANpNv9W0','KuZ-ZU8o8VM','L1O4Tqtl-Ts','lHjdZiSXH6U','lixBnuPY1kU','LK-ijBpGaMY','lS_lF8qXP8A','m_1tglpJa3U','M9aYpQn107o','mA4mDgw9gZs','MFY-yVIZ04s','mFYWR8e-UTo','MGOxRGD8jVI','MsFGOJbcTmM','mzoXxaWTh9g','N5N4bHHsRKY','n7A1sSY9LQs','nFxOWWj3hpM','ng-kukZkUJ0','nHtj_mgLmEU','NsPwhY_E5pY','nUaySHRNkro','OaNTgcltRxs','oGhqerpHJ8o','Ol_-UeJqYvg','orMqgbMpOx0','oRPoMnSuZy0','p8izkBjdVoY','PSJTp8Oxygs','pSUXldFHq00','pTp2HhmoVok','pw9PaCOn13w','pwzUP7LQxW4','q8BI1I1ZFLM','Qs5dLzUh5Tg','qWZ77apWw-A','QyCJFgSTcb8','RHFqxFgy3_M','rS-xnpSlPN8','sGOaJfZ7ON0','sJ3EX6FszGE','sms_f_PYumI','snHxWQdNPow','SNWrQQSwkZ8','t1b9KwxbgJw','T8aJ5w_y6Wg','tfHqbXkJlRM','tHN4LYq_ap4','THsmZ_kXU4s','TX_a8qmqMPY','u9zGFpGeJY0','u11hu5YeiSI','U99cl9oZ9OM','Ua5yE9BuIgA','uGlm4lVoDqo','UkWYNbstUxM','UUGdULch9J8','uWOEkMaf6C0','uZaaquVFX3o','uzJbKE1ejiw','V2XnRr1RwAw','V7q_sqL1Bio','v8bmJbsgUXo','v19JaO5RPVw','VCgiuweaqS0','VfP4kzjEKoE','VPkHGd6K2pQ','VWok20BmvZc','W2NGM1m7Csg','Wg6Gwrr6bnI','wpNa8R93FJs','ww6Hv-XlkAE','wZfMDlW99Us','Xd9hGuHQi7I','XH9GtAewl6s','XNd_CpZX1xc','y-79SBEY9vA','Y9v8wmJK5lM','yANb4fFskQU','YIzu8LzuZSY','YL46jPPx84k','YrRblZnnP9c','Yz5ZizKgIKw','z7b8R9lgq8g','zNd6OOCqwbs','ZUbzSd835CA','ZUFqN8rPpD8']
all_videos_unique_long = list(set(ny_long+seattle_long+chicago_long))

filenames = all_videos_unique_long

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

for root, dirs, files in os.walk('Transcripts'):
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



train_data = []; collcted_video_id = []
for video_id in filenames:
	try:
		train_data.append(subtitle_dic[video_id])
		collcted_video_id.append(video_id)
	except:
		pass;

if load_model:
	with open(model_name, 'rb') as f:
	    vectorizer, lda_model = pickle.load(f)
	data_vectorized = vectorizer.transform(train_data)
	lda_output = lda_model.transform(data_vectorized)
else:
	vectorizer = CountVectorizer(max_df = 0.6, min_df=0.05)
	# vectorizer = TfidfVectorizer(max_df = 0.6, min_df=0.05, max_features=5000)
	data_vectorized = vectorizer.fit_transform(train_data)

	lda_model = LatentDirichletAllocation(n_components=topic_number, learning_method='online', random_state=100, max_iter=100)
	lda_output = lda_model.fit_transform(data_vectorized)


# Log Likelyhood: Higher the better
print("\nLog Likelihood: ", lda_model.score(data_vectorized))
print("Perplexity: ", lda_model.perplexity(data_vectorized))

topicnames = ["Topic " + str(i) for i in range(lda_model.n_components)]

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
# pd.set_option('display.max_columns', None)
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=12)


for i in range(0, len(topic_keywords)):
	print(topicnames[i])
	print(list(topic_keywords[i]))

if save_to_model:
	with open(save_model_name, 'wb') as f:
	    pickle.dump((vectorizer, lda_model), f)


dominant_topic = np.argmax(lda_output, axis=1)

print(Counter(dominant_topic))
exit()

lda_dic = {}

for i in range(0, len(collcted_video_id)):
	lda_dic[collcted_video_id[i]] = dominant_topic[i]

# print(lda_dic)

data_all = read_csv('all_tmp.csv')

daily_date = {}

for date_tmp in data_all[1:]:
	tmp_2 = date_tmp[1]
	daily_date[tmp_2] = []

for date_tmp in data_all[1:]:
	tmp_1 = date_tmp[0].split('=')[-1]
	tmp_2 = date_tmp[1]
	# data_noldus[tmp_1] = tmp_2
	if ((tmp_1 in filenames) and (tmp_1 in collcted_video_id)):
		daily_date[tmp_2].append(tmp_1)


# print(daily_date)

daily_features = {}
for key in daily_date:
	tmp = []
	for file in daily_date[key]:
		tmp.append(lda_dic[file])
	if (tmp!= []):
		counter_tmp = Counter(tmp)
		total_tmp = float(len(tmp))
		feature_tmp = []
		for i in range(0, 6):
			feature_tmp.append((counter_tmp[i]/total_tmp)*100)
		daily_features[key] = feature_tmp
	else:
		daily_features[key] = [0, 0, 0, 0, 0, 0]

print(daily_features)
exit()


color_list = ['c', 'g', 'r', 'm', 'y', 'k']
topic_list = [1, 3, 4]
for m in range(0, len(topic_list)):
	feature_plt = []
	i = topic_list[m]
	for key in daily_features:
		if (daily_features[key] != [0, 0, 0, 0, 0, 0]):
			feature_plt.append(daily_features[key][i])
		else:
			feature_plt.append('NaN')
	moving_avg = []
	for j in range(0, len(feature_plt)-4):
		moving_avg_ = [feature_plt[j], feature_plt[j+1], feature_plt[j+2],  feature_plt[j+3], feature_plt[j+4]]
		moving_avg_tmp = [tmp for tmp in moving_avg_ if (tmp != 'NaN')]
		moving_avg.append(sum(moving_avg_tmp)/len(moving_avg_tmp))

	x = range(5, len(feature_plt)+1)
	plt.plot(x, moving_avg, color_list[i]+'--', label='topic '+str(i))

plt.xlim([1, len(feature_plt)+1])
plt.xticks(np.arange(1, max(x)+1, 7.0))
plt.legend()
# plt.show()
plt.savefig('ny_videos_selected_topics.png')




