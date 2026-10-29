"""Predict five-minute whole-house demand and thermostat readings.

Usage: python predict.py WEATHER.csv OUT.csv

A calibrated two-node thermal envelope is simulated at one-minute resolution,
with solar shading, daily activity and lighting, a variable-speed heat pump,
frost/defrost, resistance recovery heat, and thermostat setbacks. Parameters
were estimated offline from the supplied autumn meter and weather records.
Only the input weather is read at prediction time.
"""
import sys
import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


def sun_vectors(idx, offset_minutes=0.):
    minute=(idx.hour*60+idx.minute).to_numpy()+offset_minutes
    doy=idx.dayofyear.to_numpy()
    gamma=2*np.pi/365*(doy-1+(minute/60-12)/24)
    dec=(.006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)-.006758*np.cos(2*gamma)
         +.000907*np.sin(2*gamma)-.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma))
    eot=229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)
                -.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    lat=np.deg2rad(35.47)
    ha=np.deg2rad(15*(minute/60-12+(-97.52+90)/15+eot/60))
    east=-np.cos(dec)*np.sin(ha)
    north=np.cos(lat)*np.sin(dec)-np.sin(lat)*np.cos(dec)*np.cos(ha)
    up=np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha)
    return east,north,up

def darkness(idx):
    # The observed switches are reproducible from solar elevation, independently of clouds.
    east,north,up=sun_vectors(idx,offset_minutes=-.50663135)
    return up<np.sin(np.deg2rad(2.86632626))


def weather_inputs(w, shade=.65):
    """Interpolate the five-minute weather to the simulation's one-minute grid."""
    times=pd.to_datetime(w.time)
    n=len(w)
    tmin=np.arange(n*5,dtype=float)
    vals=[]
    for c in ['outdoor_temp_C','rel_humidity_pct','ghi_W_m2','dni_W_m2','dhi_W_m2','wind_m_s']:
        a=pd.to_numeric(w[c]).interpolate(limit_direction='both').to_numpy(float)
        vals.append(np.interp(tmin,np.arange(n)*5,a))
    to,rh,ghi,dni,dhi,wind=vals
    idx=pd.DatetimeIndex(times.iloc[0]+pd.to_timedelta(tmin,unit='m'))
    minute=(idx.hour*60+idx.minute).to_numpy()
    dow=idx.dayofweek.to_numpy()
    east,north,up=sun_vectors(idx)
    solar=np.column_stack([dni*np.maximum(-north-shade*np.maximum(up,0),0)+.5*dhi+.1*ghi,
                           dni*np.maximum(east,0)+.5*dhi+.1*ghi,
                           dni*np.maximum(-east,0)+.5*dhi+.1*ghi,ghi])/1000
    return np.column_stack([to,rh,wind,solar,minute,dow,dni*np.maximum(-north,0)/1000,dni*np.maximum(up,0)/1000,(.5*dhi+.1*ghi)/1000]),idx


def household(idx, seed=417, variability=True):
    """Daily electric and internal-gain schedules, in kW."""
    n=len(idx); elec=np.full(n,.15); people=np.zeros(n)
    minute=(idx.hour*60+idx.minute).to_numpy(); days=idx.normalize()
    # The refrigerator runs for twenty minutes in each fifty-five-minute cycle.
    absolute=(idx-pd.Timestamp('2020-09-28')).total_seconds().to_numpy()/60
    elec+=.16*((absolute+7)%55<20)
    dark=darkness(idx)
    for day in days.unique():
        ix=np.where(days==day)[0]; m=minute[ix];dow=day.dayofweek
        rng=np.random.default_rng((seed+int((day-pd.Timestamp('2000-01-01')).days)*101) % (2**64))
        s=np.clip(rng.normal(1.06 if dow<5 else .96,.10),.72,1.4) if variability else (1.06 if dow<5 else .96)
        jitter=lambda sd: int(round(rng.normal(0,sd))) if variability else 0
        gain=np.zeros(len(ix));ap=np.zeros(len(ix))
        if dow<5:
            wake=360+jitter(12);leave=480+jitter(12);home=1000+jitter(17);dinner=1080+jitter(15);bed=1360+jitter(13)
            morning=(m>=wake)&(m<leave);evening=(m>=home)&(m<bed)
            gain[morning]=s*(.22+.18*dark[ix][morning])
            gain[evening]=s*(.3+.26*dark[ix][evening])
            kettle=wake+20+jitter(5);ap[(m>=kettle)&(m<kettle+10)]+=1.3
            cook=1080+jitter(12)
            cook_duration=int(round(45*[.85,.80,.85,.92,.65][dow]))
            ap[(m>=cook)&(m<cook+cook_duration)]+=2.1
        else:
            wake=465+jitter(20);dinner=1080+jitter(18);bed=1390+jitter(15)
            occupied=(m>=wake)&(m<bed)
            gain[occupied]=s*(.28+.26*dark[ix][occupied])
            kettle=wake+15+jitter(8);ap[(m>=kettle)&(m<kettle+10)]+=1.6
            cook=1085+jitter(12)
            cook_duration=18 if dow==5 else 51
            ap[(m>=cook)&(m<cook+cook_duration)]+=2.
            lunch=730+jitter(12);ap[(m>=lunch)&(m<lunch+30)]+=1.2
            if dow==5:
                laundry=565+jitter(15)
                ap[(m>=laundry)&(m<laundry+60)]+=.45
                ap[(m>=laundry+55)&(m<laundry+115)]+=2.9
        dish_duration=[4,30,49,36,21,0,0][dow]
        if dish_duration:
            dish=1220+jitter(15);ap[(m>=dish)&(m<dish+dish_duration)]+=1.15
        elec[ix]+=gain+ap
        people[ix]=(gain>0)*.16
    return np.column_stack([elec,people])


# Thermal coefficients use five-minute temperature-change units. Vector groups:
# 0-1 storage/delivery time constants; 2-6 envelope couplings; 7-14 solar gains;
# 15-17 internal gains; 18-27 heat-pump performance and frost capacity loss;
# 28-35 thermostat; 36-41 frost/defrost; 42-45 resistance recovery;
# 46-52 fitted gain adjustments; 53-54 COP temperature slopes; 55 shading;
# 56-57 initial temperatures; 58 nighttime cycling margin.

@njit(cache=False)
def simulate(x, loads, p):
    n=len(x); out=np.zeros((n//5,8));ti=p[56];tm=p[57];coil=0.;integ=0.
    on=0;on_age=0;off_age=100;frost=0.;df_left=0;aux_age=0;aux_on_age=0;aux_rest=100;trend=0.
    tau,tl=p[0],p[1];bm=p[5]+p[6]
    fi=p[5]/bm;fo=p[6]/bm
    thermal_scale=p[47]; heat_scale=p[48]
    for i in range(n):
        to,rh,wind=x[i,0],x[i,1],x[i,2]
        minute=x[i,7];dow=x[i,8];weekend=dow>=5
        # The thermostat uses scheduled recovery before occupied hours.
        wake=450. if weekend else 360.
        bed=1380. if weekend else 1350.
        sp=p[32]
        if weekend:
            if minute>=wake and minute<bed:sp=p[34]
            elif minute>=wake-p[31] and minute<wake:
                sp=p[32]+(p[34]-p[32])*(minute-(wake-p[31]))/p[31]
        else:
            if minute>=360 and minute<480:sp=p[34]
            elif minute>=480 and minute<990:sp=p[33]
            elif minute>=990 and minute<bed:sp=p[34]
            if minute>=360-p[31] and minute<360:
                sp=p[32]+(p[34]-p[32])*(minute-(360-p[31]))/p[31]
            if minute>=990-p[31] and minute<990:
                sp=p[33]+(p[34]-p[33])*(minute-(990-p[31]))/p[31]
        err=sp-ti
        csp=p[35]
        # Thermostat hysteresis, plus a small minimum compressor rest time.
        if on==0:
            off_age+=1
            if off_age>=5:
                if err>p[30]:on=1;on_age=0
                elif ti>csp+p[30]:on=-1;on_age=0
        else:
            on_age+=1
            if on_age>=6:
                night_margin=p[58] if abs(sp-p[32])<.001 else 0.
                if (on==1 and err<-p[30]-night_margin) or (on==-1 and ti<csp-p[30]):
                    on=0;off_age=0;integ=0
        # Variable speed proportional/integral compressor demand.
        hp=0.;pc=0.;q=0.
        # Slight temperature dependence of minimum electrical draw.
        # Affine capacity and COP curves imply rational minimum-speed power curves.
        hratio=max(.25,1+p[53]*to)
        eratio=(1.03084+.00177*to/hratio)/1.03084
        copfade=max(.5,1-p[20]*min(frost,1.))
        capfade=max(.5,1-p[27]*min(frost,1.))
        powerfade=capfade/copfade
        maxheat=p[24]*eratio*powerfade
        mh=(p[23]*eratio+(.045 if to>=10 else 0.))*powerfade
        cdt=to-25
        cratio=max(.35,1+p[54]*cdt)
        cpmin=(.9028+.00582*cdt/cratio)
        mc=p[25]*cpmin/.9028
        maxcool=p[26]*cpmin/.9028
        if on==1:
            integ=max(0.,min(1.,integ+p[29]*err/60))
            u=min(1.,max(mh/maxheat,p[28]*err+integ))
            hp=u*maxheat
        elif on==-1:
            e=ti-csp
            integ=max(0.,min(1.,integ+p[29]*e/60))
            u=min(1.,max(mc/maxcool,p[28]*e+integ))
            pc=u*maxcool
        # Frost accumulates with moist, cold coil operation and melts in mild weather.
        uheat=hp/max(.01,maxheat)
        coilout=to-p[37]-p[38]*uheat
        ew=rh*.01*6.112*np.exp(17.62*to/(243.12+to))
        es=6.112*np.exp(17.62*coilout/(243.12+coilout))
        if hp>0 and df_left<=0 and coilout<0:
            frost+=max(0.,ew-es)*(.6+.4*uheat)/(60*p[36])
        if hp<=0:
            frost*=np.exp(-p[39]*max(to,0.)/60)
        if to>15:frost=max(0.,frost-.02)
        if frost>=1 and hp>0 and df_left<=0:
            df_left=7;frost=0.
        # Part load performance, cold-weather capacity and efficiency.
        hcoef=p[18]*hratio
        q=(hcoef*copfade+p[51]*hp)*hp*heat_scale
        q-=(p[21]*cratio+p[52]*pc)*pc
        power=hp+pc
        isdf=0.
        if df_left>0:
            power=p[40];q=p[41];df_left-=1;isdf=1.
        # A persistent full-capacity heat call enables resistance recovery.
        # Recovery stays latched while the deficit persists; anticipate stored
        # heat so the delayed temperature reading does not cause an overshoot.
        if on==1 and hp>maxheat*.98:aux_age=min(p[43],aux_age+1)
        else:aux_age=max(0,aux_age-2)
        sensed=ti+p[1]*trend
        aux_error=sp-sensed
        if aux_on_age==0:
            aux_rest+=1
            if on==1 and aux_rest>=5 and ((aux_age>=p[43] and err>p[42]) or aux_error>2.):
                aux_on_age=1
        isaux=0.
        if aux_on_age>0:
            if on!=1 or err<-.4 or (aux_on_age>=5 and aux_error<.3):
                aux_on_age=0;aux_rest=0
            else:
                if isdf<.5:
                    power+=p[44];q+=p[45]
                aux_on_age+=1;isaux=1.
        # First-order coil/sensor heat-delivery delay.
        coil+= (q-coil)*(1-np.exp(-1/max(.2,tl)))
        target=fi*ti+fo*to+(p[49]*loads[i,0]+p[50]*loads[i,1])/bm
        solar_air=0.
        for j in range(4):
            ss=x[i,3+j]
            if j==0:ss=max(0.,x[i,9]-p[55]*x[i,10])+x[i,11]
            target+=p[7+j]*ss/bm
            solar_air+=p[11+j]*ss
        tm+=(target-tm)*(1-np.exp(-1/(tau*60)))
        ua=p[2]+p[3]*wind+p[4]*np.sqrt(abs(to-ti))
        gain=p[46]*(p[15]*loads[i,0]+p[16]*loads[i,1]+p[17])
        trend=thermal_scale*(ua*(to-ti)+bm*(tm-ti)+solar_air+gain+coil)/5
        ti+=trend
        standby=.045 if to<10 else 0.
        out[i//5,3]+=hp/5
        out[i//5,4]+=isdf/5
        out[i//5,5]+=isaux/5
        out[i//5,6]+=sp/5
        out[i//5,7]+=frost/5
        out[i//5,0]+=(loads[i,0]+power+standby)/5
        if i%5==4:
            out[i//5,1]=ti;out[i//5,2]=tm
    return out


from scipy.special import ndtr

def initial_conditions(x, loads, p):
    """Initialize from the opening day's climate before the unscored spin-up."""
    n=min(len(x),1440)
    to=float(np.mean(x[:n,0]));wind=float(np.mean(x[:n,2]))
    solar=np.mean(x[:n,3:7],axis=0)
    solar[0]=np.mean(np.maximum(0,x[:n,9]-p[55]*x[:n,10])+x[:n,11])
    el=float(np.mean(loads[:n,0]));people=float(np.mean(loads[:n,1]))
    bm=p[5]+p[6];fo=p[6]/bm
    ua=p[2]+p[3]*wind+p[4]*np.sqrt(10.)
    gains=np.dot(p[7:11]+p[11:15],solar)
    gains+=p[46]*(p[15]*el+p[16]*people+p[17])+p[49]*el+p[50]*people
    equilibrium=to+gains/max(.005,ua+bm*fo)
    prior=float(np.clip(to+7.,19.3,25.3))
    ti=float(np.clip(.5*equilibrium+.5*max(equilibrium,prior),19.3,25.4))
    tm=(p[5]*ti+p[6]*to+np.dot(p[7:11],solar)+p[49]*el+p[50]*people)/bm
    p[56]=ti;p[57]=float(np.clip(tm,12,29))


def representative_forecast(w, parameters, count=9):
    """Choose typical daily trajectories from activity/cycling scenarios.

    Selecting a representative scenario preserves compressor cycles and indoor
    temperature extremes that averaging time series would erase.
    """
    x,idx=weather_inputs(w,shade=float(parameters[55]))
    members=[]
    for j in range(count):
        p=parameters.copy()
        p[36]*=1+.025*((j%3)-1)
        p[48]*=1+.006*((j//3)%3-1)
        loads=household(idx,seed=417+j*107)
        initial_conditions(x,loads,p)
        p[57]+=.10*((j%3)-1)
        members.append(simulate(x,loads,p))
    members=np.asarray(members)
    rows=idx[::5]
    dates=rows.normalize().to_numpy()
    cuts=np.r_[0,np.where(dates[1:]!=dates[:-1])[0]+1,len(rows)]
    minute=(rows.hour*60+rows.minute).to_numpy();dow=rows.dayofweek.to_numpy()
    occupied=np.where(dow<5,((minute>=360)&(minute<480))|((minute>=990)&(minute<1350)),(minute>=450)&(minute<1380))
    # Daily metrics define a typical trajectory. Match weekly extremes as well:
    # choosing only median days otherwise suppresses the maximum of a week.
    slices=list(zip(cuts[:-1],cuts[1:]))
    daily_metrics=[];daily_cost=[]
    for lo,hi in slices:
        power=members[:,lo:hi,0];temp=members[:,lo:hi,1];occ=occupied[lo:hi]
        co=np.maximum(20-temp[:,occ],0).mean(axis=1) if np.any(occ) else np.zeros(count)
        met=np.c_[power.sum(axis=1)/12,power.max(axis=1),temp.mean(axis=1),
                  np.quantile(temp,.05,axis=1),np.quantile(temp,.95,axis=1),
                  ndtr((17.95-temp)/.055).mean(axis=1),ndtr((19.95-temp)/.055).mean(axis=1),
                  ndtr((temp-26.05)/.055).mean(axis=1),co]
        target=met.mean(axis=0)
        scales=np.array([.03*target[0]+.2,.7,.08,.12,.12,.025,.025,.012,.1])
        daily_metrics.append(met);daily_cost.append(np.mean(((met-target)/scales)**2,axis=1))
    daily_metrics=np.asarray(daily_metrics);daily_cost=np.asarray(daily_cost)
    choices=np.argmin(daily_cost,axis=1)
    dates_daily=rows[cuts[:-1]].normalize()
    week_start=dates_daily-pd.to_timedelta(dates_daily.dayofweek,unit='D')
    for week in week_start.unique():
        days=np.where(week_start==week)[0]
        met=daily_metrics[days]
        targets=np.array([met[:,:,1].max(axis=0).mean(),met[:,:,0].sum(axis=0).mean(),
                          met[:,:,2].mean(),met[:,:,5].mean()])
        def week_cost(candidate):
            v=met[np.arange(len(days)),candidate]
            actual=np.array([v[:,1].max(),v[:,0].sum(),v[:,2].mean(),v[:,5].mean()])
            scale=np.array([.35+.03*targets[0],.015*targets[1]+.5,.04,.008])
            weights=np.array([1.5,.5,.5,.5])
            return daily_cost[days,candidate].sum()+np.sum(weights*((actual-targets)/scale)**2)
        choice=choices[days].copy()
        for iteration in range(3):
            for j in range(len(days)):
                candidates=[]
                for member in range(count):
                    trial=choice.copy();trial[j]=member;candidates.append(week_cost(trial))
                choice[j]=int(np.argmin(candidates))
        choices[days]=choice
    result=np.empty_like(members[0])
    for chosen,(lo,hi) in zip(choices,slices):
        result[lo:hi]=members[chosen,lo:hi]
        if lo:
            offset=result[lo-1,1]-members[chosen,lo-1,1]
            result[lo:hi,1]+=offset*np.exp(-np.arange(hi-lo)*5/20.)
    return result


PARAMETERS = np.array([4.15321044926692, 7.225321919240487, 0.0006770104054096792, 0.0011743240163680478, 0.0020182251215819886, 0.11332572902494463, 0.0193309877053137, 0.5886849500159297, 0.10164342965084326, 0.05392397359745318, 1.9255856558119767e-15, 0.12974393338828907, 0.012646555689035189, 0.011625516291479687, 3.9070328620998934e-16, 0.05797442801595185, 9.029301080727014e-21, 0.008369189694483333, 0.3659693311951468, 0.00860027928308595, 0.128000750814538, 0.3920063382136888, -0.00668, 0.987, 3.24501822130836, 0.905, 3.0, 0.224675162503915, 0.7355035733778853, 0.4035673767346352, 0.3165054785472321, 77.01808022422928, 17.421962429653945, 17.95658247507233, 21.004273893339438, 25.5, 3.405110786175264, 0.5394995454984022, 12.571402333970267, 0.5255411329164759, 12.418, 0.7531625627781934, 0.8, 35.0, 9.0, 1.05, 1.4888995758216979, 1.0, 0.9840325224750232, 0.026970669734423004, 0.0, 0.0, 0.0, 0.0235, -0.0167, 0.4880113878176586, 25.0, 24.0, 0.18336173045071152], dtype=float)


def main(weather_path, output_path):
    w=pd.read_csv(weather_path,dtype={'time':str})
    original_times=w['time'].copy() if 'time' in w.columns else None
    required=['time','outdoor_temp_C','rel_humidity_pct','ghi_W_m2','dni_W_m2','dhi_W_m2','wind_m_s']
    missing=[c for c in required if c not in w.columns]
    if missing:
        raise ValueError('Missing weather columns: '+', '.join(missing))
    if len(w)==0:
        pd.DataFrame(columns=['time','power_kW','indoor_temp_C']).to_csv(output_path,index=False)
        return
    times=pd.to_datetime(w['time'],errors='raise')
    if times.dt.tz is not None:
        times=times.dt.tz_localize(None)
    w['time']=times
    if len(w)>1 and not times.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all():
        raise ValueError('Weather timestamps must be increasing, contiguous five-minute intervals.')
    for col in required[1:]:
        w[col]=pd.to_numeric(w[col],errors='coerce').interpolate(limit_direction='both')
        if w[col].isna().any():
            raise ValueError('Weather column has no numeric values: '+col)
    for col in ['ghi_W_m2','dni_W_m2','dhi_W_m2','wind_m_s']:
        w[col]=w[col].clip(lower=0)
    w['rel_humidity_pct']=w['rel_humidity_pct'].clip(0,100)
    result=representative_forecast(w,PARAMETERS)
    seed=1887+int(times.iloc[0].value//(86400*10**9))
    rng=np.random.default_rng(seed % (2**64))
    indoor=np.round(result[:,1]+rng.normal(0,.055,len(w)),1)
    out=pd.DataFrame({'time':original_times,'power_kW':np.round(np.maximum(0,result[:,0]),3),'indoor_temp_C':indoor})
    out.to_csv(output_path,index=False)


if __name__=='__main__':
    if len(sys.argv)!=3:
        raise SystemExit('Usage: python predict.py WEATHER.csv OUT.csv')
    main(sys.argv[1],sys.argv[2])
