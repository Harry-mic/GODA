import numpy as np
import pandas as pd
import pdb
#s6~s10
s6s10 = pd.read_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states\s6~s10.csv")
s6s10  = np.array(s6s10)

exact_s6s10 = s6s10[0:5,:]*s6s10[6,:]+s6s10[5,:]
exact_s6s10 = pd.DataFrame(exact_s6s10)
exact_s6s10.to_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states\exact_s6~s10.csv")

#s6'~s10'
s6_s10_ = pd.read_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states\s6'~s10'.csv")
s6_s10_ = np.array(s6_s10_)

exact_s6_s10_ = s6_s10_[0:5,:]*s6_s10_[6,:]+s6_s10_[5,:]
exact_s6_s10_ = pd.DataFrame(exact_s6_s10_)
exact_s6_s10_.to_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states\exact_s6'~s10'.csv")

#s6''~s10''
s6__s10__ = pd.read_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states\s6''~s10''.csv")
s6__s10__ = np.array(s6__s10__)

exact_s6__s10__ = s6__s10__[0:5,:]*s6__s10__[6,:]+s6__s10__[5,:]
exact_s6__s10__ = pd.DataFrame(exact_s6__s10__)
exact_s6__s10__.to_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states\exact_s6''~s10''.csv")