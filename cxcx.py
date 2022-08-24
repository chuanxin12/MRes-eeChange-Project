import numpy as np
from typing import Optional, Union
from typing_extensions import Self
from zipfile import ZIP_BZIP2
import numpy as np
from warnings import warn
from pyrealm import pmodel
from pyrealm.utilities import summarize_attrs
from pyrealm.param_classes import PModelParams
# from pyrealm.boudspyrealmnds_checker import bounds_checker
from pyrealm.utilities import check_input_shapes


from pyrealm.pmodel import PModelEnvironment, PModel 
from pyrealm.pmodel import CalcOptimalChi
from pyrealm.pmodel import CalcLUEVcmax


class CalcF:
    def __init__(self,
                env: PModelEnvironment,
                pmo1: PModel,# 所以运行F的时候要先运行pmo1,运行pmo1之前先运行env #与do_soilmstress,do_ftemp_kphio有关
                maxYII: Union[float, np.ndarray]= None, 
                #NIRV: Union[float, np.ndarray]= None,
                fapar: Union[float, np.ndarray]= 1,#
                ppfd: Union[float, np.ndarray]= 300 #要改成默认 ppfd= 1 么，PModel是默认值是1  
                ):

# 0.1 Check imputs
        self.shape = check_input_shapes(pmo1.do_soilmstress,pmo1.do_ftemp_kphio,fapar,ppfd)# 待改
# 0.2 set attribute defaults  需要输入的参数 inputs都需要初始值
        self.tc = env.tc
        self.do_soilmstress = pmo1.do_soilmstress
        self.do_ftemp_kphio = pmo1.do_ftemp_kphio
        self.fapar = fapar
        self.ppfd = ppfd
        self.lue = pmo1.lue 
        self.ci = pmo1.optchi.ci
        self.gammastar = env.gammastar
        self.Kf = 0.05
        self.k_cmolmass = 12.0107 # unit g mol-1
        # self.NIRV = NIRV
#0.2 Output defults   Output的值一般设定为None
        self.maxYII = None
        self.iabs = None
        self.gpp = None
        self.J = None
        self.realYII = None
        self.Fluorescence = None
        self.PhiN = None
        self.PhiD = None
        self.PhiF = None
        self.PhiF1 = None
        self.PhiFt = None
        self.Kd = None
        self.Kn = None
        self.Kp = None 
        self.x_rc  = None 
        self.NPQ = None
# 1 maxYII defaults: 
#maxYII,namely ϕ0p, is the  maximum quantum yield efficiency(C.van der Tol 2014)
        
        ftmax = 0.7078824 # stocker eq.20                               #备注ftmax怎么算出来的 # stocker eq  20 
        if maxYII is None:      
            if not self.do_ftemp_kphio: # 情况一self.do_ftemp_kphio = False, so not temperature dependence
                self.maxYII = 0.049977*4  # 情况一 ，是恒值,即0.1999
            elif self.do_soilmstress:
                self.maxYII = 0.087182*4* ftmax  #情况二 即0.246858  #PModel(env，soilmstress = 某个值)
            else:
                self.maxYII = 0.081785*4* ftmax  #即0.23157, 是默认值PModel(env)
        else:
            self.maxYII = maxYII 

#soilmstress = pmodel.calc_soilmstress(soilm=0.4, meanalpha=0.9) soilmstress的值怎么算
#soilmstress
#pmodel.PModel(env, soilmstress=soilmstress)
#2 Iabs  
        self.iabs = self.fapar * self.ppfd #unit of ppfd should be check μmol m-2 s-1 then fapar unitless so iabs μmol m-2 s-1
#3 Lue        
# Light use efficiency (gpp per unit absorbed light)
#self.lue = (self.kphio * self.ftemp_kphio * self.optchi.mj * self.mjlim *self.pmodel_params.k_c_molmass * self.soilmstress)
# 即lue =φ0 *  φ0 (T)* mj* mjlim *M_c*beta,  其中 φ0 is kphio, φ0 (T) is ftemp_kphio, beta is soilmstress 


# 4 GPP
        self.gpp = self.lue * self.iabs  # 
# 5 J 
        
        self.J = 4*self.gpp/self.k_cmolmass*(self.ci+2*self.gammastar)/(self.ci-self.gammastar) # Giulia 流程图最后那个 Aj namely gpp=J/4*(ci-gammastar)/(ci+2gammastar)
        
# the real quantum yield efficiency (realYII,namely ϕp)
        self.realYII =self.J/self.iabs # C van der Tol eq.10?

# define attributes populated by chris

    # def Chris(self): # change to method 
        #x_rc  
        self.x_rc = 1 - self.realYII/self.maxYII  #eq.15 in  C.van der Tol 2014  
        b1 = 0.114
        a1 = 2.83
        Kn0 = 2.48
        v = ((1+b1)*(self.x_rc**a1))/(b1 + self.x_rc**a1)

        #Rates coefficient(K) 
        # self.tc = np.asarray(env.tc)
        # self.Kd = max(0.03 * env.tc + 0.0773, 0.87) 
        kd = 0.03 * env.tc + 0.0773
        kd = np.asarray(kd)
        kd[kd<0.87] = 0.87
        self.Kd = kd
         #if self.Kd is None:
             #if kd >= 0.87:
                 #self.Kd = kd 
             #else:
                 #self.Kd = 0.87

        #原版self.Kd = max(0.03 * env.tc + 0.0773, 0.87) 
        #实验 self.Kd = 0.03 * env.tc 

        self.Kn = v*Kn0 #eq 19
        self.Kf = 0.05 #问catherine Kf 是等于0.05么，备注里没提到Kf 
        self.Kp = (self.realYII*(self.Kd+self.Kn+self.Kf))/(1-self.realYII)#eq2
        
        #Yields
        self.PhiN = self.Kn/(self.Kd+self.Kn+self.Kf+self.Kp)
        self.PhiD = self.Kd/(self.Kd+self.Kn+self.Kf+self.Kp)
        self.PhiF = 1-self.realYII-self.PhiN-self.PhiD # eq7
        
        # self.PhiF1 = self.Kf/(self.Kf+self.Kd+self.Kn) #eq. 13，KP=0，so no kp in this PhiF
        # self.PhiFt = (1-self.maxYII+self.x_rc*self.maxYII)*self.PhiF1 #eq 17, maxYII

        self.Fluorescence = self.PhiF* self.iabs # Fluorescence
        self.NPQ = self.PhiN*self.iabs
        # self.SIF_NIRV = self.NIRV*self.ppfd *self.PhiF


# example 
# env = PModelEnvironment(tc=20, vpd=1000, co2=400, patm=101325.0)                                                                                
# pmo1 = PModel(env)
# d = CalcF (env=env , pmo1 = pmo1, fapar=1, ppfd=300)


# #plot  问题出在 self.Kd = max(0.03 * env.tc + 0.0773, 0.87)  

# from matplotlib import pyplot
# import numpy as np
# tc = np.linspace(20, 30, 5)
# print(tc) 
# env = PModelEnvironment(tc=tc, vpd=1000, co2=400, patm=101325.0) 
# pmo1 = PModel(env)
# d = CalcF (env=env , pmo1 = pmo1, fapar=1, ppfd=300)

# # Plot TC against Fluorescence
# pyplot.plot(tc, d.Fluorescence)
# pyplot.xlabel('Temperature °C')
# pyplot.ylabel('Fluorescence')
# pyplot.show()


# print(d.gpp)# 76.425 in grams
# print(d.J) # 35.7

# print(d.realYII) # 0.1191
# print(d.maxYII) #0.2315 
# print(d.x_rc)  #0.4856
# # self.x_rc = 1 - self.realYII/self.maxYII
# # v = ((1+b1)*(self.x_rc**2.83))/(b1 + self.x_rc**2.83)
# #self.Kn = v*2.48 #eq 19

# # for self.x_rc**2.83, x_rc namely x should be positive 

# print(d.Kn) # 1.469


# print(d.Fluorescence) # 5.530
# print(d.iabs) #300
# print(d.PhiF) # 0.01843
# print(d.realYII) # 0.119
# print(d.PhiN) #0.5417
# print(d.PhiD ) #0.3207

# print(d.Kf) # 0.05
# print(d.Kd) # 0.87
# print(d.Kn) # 1.46937
# print(d.Kp) # 0.3230


# def Chris(self): #改成method 


