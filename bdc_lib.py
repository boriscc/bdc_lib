import numpy as np
from scipy.special import gammaln

def get_num_with_err(num, err_up, err_down = None, num_significant = float('inf'), err_significant = 2):
    """Returns a string representation of a number accounting for uncertainty"""

    def get_level(x):
        x = np.abs(x)
        if(x == 0): level = 0
        elif(x < 1): level = np.floor(np.log10(x))
        else: level = np.floor(np.log10(x))
        return level

    def get_disp(significant, err, num):
        expo = np.log10(10**significant / err)
        if(expo == np.floor(expo)):
            expo = np.floor(expo) - 1
        else:
            expo = np.floor(expo)
        err_disp = round(err*10**expo)
        num_disp = round(num*10**expo)*10**(-expo)
        return expo, err_disp, num_disp

    if(err_down == None): err_down = -err_up
    sym = 0
    if -err_down == err_up: sym = 1
    assert isinstance(num, float)
    assert isinstance(err_up, float)
    assert isinstance(err_down, float)
    assert isinstance(num_significant, float) or isinstance(num_significant, int)
    assert isinstance(err_significant, int)
    
    assert err_down <= 0 and err_up >= 0, "Bad error limits"

    significant = min([get_level(max([-err_down, err_up])) - get_level(num) + num_significant, err_significant])

    if significant <= -1:
        level = get_level(num)
        num = 10**(level+1-num_significant)*round(num*10**(num_significant-level-1))
        if level+1-num_significant >= 0:
            str = '%+.0f' % num
        else:
            form = '%%+.%df' % (num_significant - level - 1)
            str = form % num
        return str
    
    if sym == 1:
        err = err_up
        assert err > 0, 'Bad error' 

        expo, err_disp, num_disp = get_disp(significant, err, num)
        if get_level(err_disp) == err_significant:
            significant = significant - 1
            expo, err_disp, num_disp = get_disp(significant, err, num)
        if expo >= 0:
            if err_disp == 0:
                form = '%%+.%df' % expo
                str = form % num_disp
            else:
                form = '%%+.%df(%%d)' % expo
                str = form % (num_disp, err_disp)
        else:
            if err_disp == 0:
                str = '%+*.0f' % (significant - expo, num_disp)
            else:
                str = '%+*.0f(%d)' % (significant - expo, num_disp, err_disp*10**(-expo))
    else:
        assert err_up > err_down, 'Must have one error with |err| > 0'
        if err_up == 0:
            expo, err_disp, num_disp = get_disp(significant, -err_down, num)
            if get_level(err_disp) == err_significant:
                significant = significant - 1
                expo, err_disp, num_disp = get_disp(significant, -err_down, num)
            if expo >= 0:
                if err_disp == 0:
                    form = '%%+.%df' % expo
                    str = form % num_disp
                else:
                    form = '%%+.%df_{(-%%d)}^{(+0)}' % expo
                    str = form % (num_disp, err_disp)
            else:
                if err_disp == 0:
                    str = '%+*.0f' % (significant - expo, num_disp)
                else:
                    str = '%+*.0f_{(-%d)}^{(+0)}' % (significant - expo, num_disp, err_disp*10**(-expo))
        elif err_down == 0:
            expo, err_disp, num_disp = get_disp(significant, err_up, num)
            if get_level(err_disp) == err_significant:
                significant = significant - 1
                expo, err_disp, num_disp = get_disp(significant, err_up, num)
            if expo >= 0:
                if err_disp == 0:
                    form = '%%+.%df' % expo
                    str = form % num_disp
                else:
                    form = '%%+.%df_{(-0)}^{(+%%d)}' % expo
                    str = form % (num_disp, err_disp)
            else:
                if err_disp == 0:
                    str = '%+*.0f' % (significant - expo, num_disp)
                else:
                    str = '%+*.0f_{(-0)}^{(+%d)}' % (significant - expo, num_disp, err_disp*10**(-expo))
        else:
            if -err_down < err_up:
                err_max = err_up
            else:
                err_max = -err_down
            expo, err_disp, num_disp = get_disp(significant, err_max, num)
            if get_level(err_disp) == err_significant:
                significant = significant - 1
                expo, err_disp, num_disp = get_disp(significant, err_max, num)
            if -err_down > err_up:
                err_down_disp = err_disp
                err_up_disp = round(err_up*10**expo)
            else:
                err_up_disp = err_disp
                err_down_disp = -round(err_down*10**expo)
            if expo >= 0:
                if err_up_disp == 0 and err_down_disp == 0:
                    form = '%%+.%df' % expo
                    str = form % num_disp
                elif err_up_disp == err_down_disp:
                    form = '%%+.%df(%%d)' % expo
                    str = form % (num_disp, err_up_disp)
                else:
                    form = '%%+.%df_{(-%%d)}^{(+%%d)}' % expo
                    str = form % (num_disp, err_down_disp, err_up_disp)
            else:
                if err_up_disp == 0 and err_down_disp == 0:
                    str = '%+*.0f' % (significant - expo, num_disp)
                elif err_up_disp == err_down_disp:
                    str = '%+*.0f(%d)' % (significant - expo, num_disp, err_up_disp*10**(-expo))
                else:
                    str = '%+*.0f_{(-%d)}^{(+%d)}' % (significant - expo, num_disp, err_down_disp*10**(-expo), err_up_disp*10**(-expo))
    return str

class sample_stat_norm:
    """Class with statistical info"""
    def __init__(self, x):
        self.n = float(len(x))

        self.m = np.mean(x)
        self.m2 = sum((x-self.m)**2) / self.n
        self.m3 = sum((x-self.m)**3) / self.n
        self.m4 = sum((x-self.m)**4) / self.n
        self.k1 = self.m
        self.k2 = self.n * self.m2 / (self.n-1)
        self.k3 = self.m3 * self.n**2/((self.n-1)*(self.n-2))
        self.k4 = self.n**2 * ((self.n+1)*self.m4 - 3*(self.n-1)*self.m2**2) / ((self.n-1)*(self.n-2)*(self.n-3))
        K = np.sqrt((self.n-1)/2)*np.exp(gammaln(0.5*self.n-0.5) - gammaln(0.5*self.n))
        
        # Sample stats
        # Definition
        self.sample_mean = self.m
        # Definition
        self.sample_var = self.m2
        # Definition
        self.sample_stddev = np.sqrt(self.sample_var)

        # Pop estimators
        # Unbiased, valid for all parent dists
        self.est_pop_mean = self.m
        self.est_pop_var = self.k2
        # See e.g. www.uic.edu/classes/idsc/ids571/samplvar.pdf
        # K = 1 / c_4, valid for norm, unbiased
        self.est_pop_stddev = K * np.sqrt(self.est_pop_var)

        # Stddev and var estimators for Sample stats
        self.est_sample_mean_stddev = self.est_pop_stddev / np.sqrt(self.n)
        self.est_sample_mean_var = self.est_pop_var / self.n
        self.est_sample_stddev_stddev = self.sample_stddev * np.sqrt(K*K-1)
        self.est_sample_stddev_var = self.est_pop_var * (self.n-1) * (1 - K**(-2)) / self.n

        # Stddev and var estimators for Pop estimators
        self.est_pop_mean_stddev = self.est_pop_stddev / np.sqrt(self.n)
        self.est_pop_mean_var = self.est_pop_var / self.n
        self.est_pop_stddev_stddev = self.est_pop_stddev * np.sqrt(K*K-1)
        self.est_pop_stddev_var = self.est_pop_var * (K**2 - 1)
        
        # Standard deviation of sqrt(m2) = sqrt(m2) * sqrt(K**2 - 1)
        # See e.g. stats.stackexchange.com/question/631/standard-deviation-of-standard-deviation
        # Variance of sqrt(m2) = 

        #self.pop_var_stddev = self.pop_var * np.sqrt(2 / (self.n-1))
        #self.pop_var_var = self.pop_var_stddev**2
        
        # Sample skewness
        # Definition
        self.sample_skew = self.k3 / self.k2**1.5
        self.est_sample_skew_var = 6*self.n*(self.n-1) / ((self.n-2)*(self.n+1)*(self.n+3))
        self.est_sample_skew_stddev = np.sqrt(self.est_sample_skew_var)

        # Population skewness
        self.est_pop_skew = self.m3 / self.k2**1.5
        self.est_pop_skew_var = self.est_sample_skew_var # Incorrect, pop_skew_var < sample_skew_var so I use this as approx
        self.est_pop_skew_stddev = np.sqrt(self.est_pop_skew_var)

        # Sample excess kurtosis
        # Definition
        self.sample_exkurt = self.m4 / self.m2**2 - 3
        self.est_sample_exkurt_var = 24*self.n*(self.n-1)**2 / ((self.n-3)*(self.n-2)*(self.n+3)*(self.n+5))
        self.est_sample_exkurt_stddev = np.sqrt(self.est_sample_exkurt_var)

        # Unbiased (in the norm case) estimator of the population excess kurtosis
        self.est_pop_exkurt = self.k4 / self.est_pop_var**2
        self.est_pop_exkurt_var = self.est_sample_exkurt_var # Incorrect, I use this as an approximation
        self.est_pop_exkurt_stddev = np.sqrt(self.est_sample_exkurt_var)

def phys_const():
    r1 = 'CODATA 2010 (http://arxiv.org/abs/1203.5425)';
    
    c.c = 299792458;
    c.c_unit = 'm/s';
    c.c_err = 0;
    c.c_SI = c.c;
    c.c_unit_SI = c.c_unit;
    c.c_err_SI = 0;
    c.c_ref = r1;
    
    c.hbarc = 197.3269718;
    c.bharc_unit = 'Mev*fm';
    c.hbarc_err = 0.0000044;
    c.hbarc_ref = r1;
    
    c.alpha = 0.0072973525698;
    c.alpha_unit = '';
    c.alpha_err = 0.0000000000024;
    c.alpha_SI = c.alpha;
    c.alpha_unit_SI = c.alpha_unit;
    c.alpha_err_SI = c.alpha_err;
    c.alpha_ref = r1;
    
    c.mp = 938.272046;
    c.mp_unit = 'MeV';
    c.mp_err = 0.000021;
    c.mp_SI = 1.672621777e-27;
    c.mp_unit_SI = 'kg';
    c.mp_err_SI = 0.000000074e-27;
    c.mp_ref = r1;

    c.mn = 939.565379;
    c.mn_unit = 'MeV';
    c.mn_err = 0.000021;
    c.mn_SI = 1.674927351e-27;
    c.mn_unit_SI = 'kg';
    c.mn_err_SI = 0.000000074e-27;
    c.mn_ref = r1;
    
    c.md = 1875.612859;
    c.md_unit = 'MeV';
    c.md_err = 0.000041;
    c.md_SI = 3.34358348e-27;
    c.md_unit_SI = 'kg';
    c.md_err_SI = 0.00000015e-27;
    c.md_ref = r1;
    
    c.mt = 2808.921005;
    c.mt_unit = 'MeV';
    c.mt_err = 0.000062;
    c.mt_SI = 5.00735630e-27;
    c.mt_unit_SI = 'kg';
    c.mt_err_SI = 0.00000022e-27;
    c.mt_ref = r1;
    
    c.mh = 2808.391482;
    c.mh_unit = 'MeV';
    c.mh_err = 0.000062;
    c.mh_SI = 5.00641234e-27;
    c.mh_unit_SI = 'kg';
    c.mh_err_SI = 0.00000022e-27;
    c.mh_ref = r1;
    
    c.mHe = 3727.379240;
    c.mHe_unit = 'MeV';
    c.mHe_err = 0.000082;
    c.mHe_SI = 6.64465675e-27;
    c.mHe_unit_SI = 'kg';
    c.mHe_err_SI = 0.00000029e-27;
    c.mHe_ref = r1;
    
    c.mn_over_mp = 1.00137841917;
    c.mn_over_mp_unit = '';
    c.mn_over_mp_err = 0.00000000045;
    c.mn_over_mp_unit_SI = '';
    c.mn_over_mp_err_SI = c.mn_over_mp_err;
    c.mn_over_mp_ref = r1;
    
    c.md_over_mp = 1.99900750097;
    c.md_over_mp_unit = '';
    c.md_over_mp_err = 0.00000000018;
    c.md_over_mp_unit_SI = '';
    c.md_over_mp_err_SI = c.md_over_mp_err;
    c.md_over_mp_ref = r1;
    
    c.mt_over_mp = 2.9937170308;
    c.mt_over_mp_unit = '';
    c.mt_over_mp_err = 0.0000000025;
    c.mt_over_mp_unit_SI = '';
    c.mt_over_mp_err_SI = c.mt_over_mp_err;
    c.mt_over_mp_ref = r1;
    
    c.mh_over_mp = 2.9931526707;
    c.mh_over_mp_unit = '';
    c.mh_over_mp_err = 0.0000000025;
    c.mh_over_mp_unit_SI = '';
    c.mh_over_mp_err_SI = c.mh_over_mp_err;
    c.mh_over_mp_ref = r1;
    
    c.mHe_over_mp = 3.97259968933;
    c.mHe_over_mp_unit = '';
    c.mHe_over_mp_err = 0.00000000036;
    c.mHe_over_mp_unit_SI = '';
    c.mHe_over_mp_err_SI = c.mHe_over_mp_err;
    c.mHe_over_mp_ref = r1;
    
    c.u = 931.494061;
    c.u_unit = 'MeV';
    c.u_err = 0.000021;
    c.u_SI = 1.660538921e-27;
    c.u_unit_SI = 'kg';
    c.u_err_SI = 0.000000073e-27;
    c.u_ref = r1;
    
    c.me = 0.510998928;
    c.me_unit = 'MeV';
    c.me_err = 0.000000011;
    c.me_SI = 9.10938291e-31;
    c.me_unit_SI = 'kg';
    c.me_err_SI = 0.00000040e-31;
    c.me_ref = r1;
    
    c.atom_ionization_E_H_1 = 13.598434005136;
    c.atom_ionization_E_H_1_unit = 'eV';
    c.atom_ionization_E_H_1_err = 0.000000000012;
    c.atom_ionization_E_H_1_ref = 'NIST ASD (http://physics.nist.gov/asd)';
    
    c.a0 = 0.52917721092;
    c.a0_unit = '�';
    c.a0_err = 0.00000000017;
    c.a0_SI = c.a0*1e-10;
    c.a0_unit_SI = 'm';
    c.a0_err_SI = c.a0_err*1e-10;
    c.a0_ref = r1;

    return c

