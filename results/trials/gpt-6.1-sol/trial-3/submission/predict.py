"""Weather-driven house and heat-pump forecast.

Usage: python predict.py WEATHER.csv OUT.csv

Calibrated on the supplied autumn meter/thermostat and weather observations.
The model integrates air, building mass, and thermostat response each minute,
with solar geometry, wind-dependent heat loss, programmed recovery, inverter
modulation, moisture accumulation/defrost, auxiliary heat, and household loads.
No fitting or access to the autumn CSV files occurs during prediction.
"""
import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


def solar_features(time, weather):
    hr = (time.hour + time.minute / 60).to_numpy()
    gamma = 2 * np.pi / 365 * (time.dayofyear.to_numpy() - 1 + (hr-12)/24)
    dec = (.006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)
           -.006758*np.cos(2*gamma)+.000907*np.sin(2*gamma)
           -.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma))
    eot = 229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)
                 -.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    ha = np.deg2rad((hr*60+eot+4*(-97.52+90))/4-180)
    lat = np.deg2rad(35.47)
    south = np.sin(lat)*np.cos(dec)*np.cos(ha)-np.cos(lat)*np.sin(dec)
    east = -np.cos(dec)*np.sin(ha)
    ghi, dni, dhi = [weather[:, k]/1000 for k in (2, 3, 4)]
    return np.array([ghi, np.maximum(south, 0)*dni+.5*dhi+.1*ghi,
                     np.maximum(east, 0)*dni+.5*dhi+.1*ghi,
                     np.maximum(-east, 0)*dni+.5*dhi+.1*ghi]).T.copy()


def make_inputs(weather, base_scale=1., random_loads=True):
    time = pd.DatetimeIndex(pd.to_datetime(weather['time']))
    # The observations are interval means; integrate equipment and temperatures
    # at one minute, retaining the temperature at the end of each interval.
    minute = pd.date_range(time[0], time[-1]+pd.Timedelta(minutes=4), freq='min')
    xp = (time.asi8-time[0].value)/60e9
    xx = np.arange(len(minute), dtype=float)
    cols = ['outdoor_temp_C','rel_humidity_pct','ghi_W_m2','dni_W_m2',
            'dhi_W_m2','wind_m_s','pressure_hPa']
    ww = np.column_stack([np.interp(xx, xp, pd.to_numeric(weather[c],errors='coerce')
                            .interpolate(limit_direction='both')) for c in cols])
    solar = solar_features(minute, ww)
    hr = (minute.hour+minute.minute/60).to_numpy()
    weekend = (minute.dayofweek >= 5).astype(np.int64)
    cls = np.where(weekend, np.where(hr < 7, 0, np.where(hr < 17, 4,
          np.where(hr < 23, 3, 0))), np.where(hr < 6, 0, np.where(hr < 8, 1,
          np.where(hr < 16, 2, np.where(hr < 22.5, 3, 0))))).astype(np.int64)
    day = minute.normalize()
    baseline = np.full(len(minute), .15)
    # Refrigerator: 20 minutes per 55-minute cycle, measured from the training
    # origin. Absolute phase makes forecasts independent of input-file slicing.
    absolute_min = ((minute.asi8-pd.Timestamp('2020-09-28').value)//60_000_000_000)
    baseline += .16*((absolute_min+7)%55 < 20)
    for date in day.unique():
        ix = np.flatnonzero(day == date)
        h = hr[ix]
        # Regular plug loads and lighting. Lighting switches at low irradiance,
        # so winter mornings and evenings differ from the autumn schedule.
        seed = int(date.value//86_400_000_000) & 0xffffffff
        rng = np.random.default_rng(seed*1009+17)
        we = date.dayofweek >= 5
        jitter = lambda scale: rng.normal(0, scale) if random_loads else 0.
        amp = lambda mean, sd: max(.05, mean+jitter(sd))
        def add(start, duration, power):
            baseline[ix] += power*((h >= start)&(h < start+duration))
        scale = max(.55, 1.03+jitter(.18))
        dark = ww[ix,2] < 20
        if we:
            awake=(h>=7.35+jitter(.12))&(h<23.15+jitter(.12))
            baseline[ix] += scale*awake*(.26+.24*dark)
        else:
            morning=(h>=6+jitter(.15))&(h<8.08+jitter(.12))
            evening=(h>=16.5+jitter(.15))&(h<22.75+jitter(.12))
            baseline[ix] += scale*(morning*(.22+.18*dark)+evening*(.30+.26*dark))
        add((8.0 if we else 6.2)+jitter(.20 if we else .12), (18 if we else 9)/60,
            1.6 if we else 1.2)
        dow=date.dayofweek
        dinner_prob=[.72,.30,.82,.82,.50,.45,.88][dow]
        dinner_duration=75 if we else 45
        if not random_loads:
            add(18.25, dinner_duration*dinner_prob/60, 2. if we else 2.1)
        elif rng.random()<dinner_prob:
            add(18.25+jitter(.16), (dinner_duration+jitter(6))/60,
                2. if we else 2.1)
        if not we:
            dish_prob=[.35,.58,.58,.40,.45][dow]
            if not random_loads:
                add(20.6, dish_prob, 1.15)
            elif rng.random()<dish_prob:
                add(20.6+jitter(.22), 1+jitter(.08), 1.15)
        if we:
            add(12.2+jitter(.15), .5, 1.2)
            if dow==5:
                add(9.1+jitter(.2), 1., .45)
                if not random_loads or rng.random()<.85:
                    add(10.3+jitter(.22), 1.25*(1 if random_loads else .85), 2.9)
    baseline = .15+(baseline-.15)*base_scale
    inp = np.column_stack([ww[:,0], ww[:,1], ww[:,5], solar,
                           hr, weekend, cls, baseline])
    return time, minute, inp


@njit
def simulate(inp, thermal, control, observed=None):
    # control: heat Kp, Ki, actuator ramp, min-on, min-off, deadband,
    # frost accumulation, capacity loss, COP loss, defrost heat, aux lockout,
    # aux delay, aux threshold, aux off, aux size, weekend ramp duration,
    # cooling efficiency, cooling temperature coefficient, heating max power.
    kp,ki,ramp,minon,minoff,db,frate,fcap,fcop,defq,auxlock,auxdelay,auxerr,auxoff,auxsize,weramp,coolh,cooltc,maxpower=control[:19]
    moisture_model=len(control)>19
    forced_defrost=inp.shape[1]>11
    a,aw,b,c,d=thermal[:5]
    sa=thermal[5:9];sm=thermal[9:13];g=thermal[13:18]
    pe=thermal[18];h0=thermal[20];h1=thermal[21];fan=thermal[22]
    sensor_tau=thermal[24] if len(thermal)>24 else 0.
    sensor_alpha=1-np.exp(-1/sensor_tau) if sensor_tau>0 else 1.
    n=len(inp);result=np.zeros((n//5, 11))
    air=23.;mass=23.;sensed=23.;integral=0.;frac=.28;mode=0;runtime=0.;offtime=30.
    frost=0.;defleft=0;aux=False;cool_integral=0.
    for i in range(n):
        to,rh,wind=inp[i,0],inp[i,1],inp[i,2]
        minfrac=max(.18,min(.4,(1.031+.0016*to-fan)/(maxpower+.005*to-fan)))
        hr=inp[i,7];we=inp[i,8]>0;cl=int(inp[i,9]);base=inp[i,10]
        occupied=control[23] if len(control)>23 else 21.
        away=control[24] if len(control)>24 else 18.
        night=control[25] if len(control)>25 else 17.5
        if we:
            if hr<6.5: sp=night
            elif hr<6.5+weramp: sp=night+(occupied-night)*(hr-6.5)/weramp
            elif hr<23: sp=occupied
            else: sp=night
        else:
            if hr<4.5: sp=night
            elif hr<6: sp=night+(occupied-night)*(hr-4.5)/1.5
            elif hr<8: sp=occupied
            elif hr<15.5: sp=away
            elif hr<17: sp=away+(occupied-away)*(hr-15.5)/1.5
            elif hr<22.5: sp=occupied
            else: sp=night
        coolsp=25.5 if sensor_tau>0 else (26. if not we and 8<=hr<16 else 25.5)
        if len(control)>26:coolsp=control[26]
        cooldb=control[27] if len(control)>27 else db
        e=sp-sensed
        if mode==1 and defleft==0 and ((e < -db and runtime>=minon) or e < -1.5):
            mode=0;offtime=0.;integral=0.;runtime=0.;aux=False
        elif mode==-1 and ((sensed<coolsp-cooldb and runtime>=minon) or sensed<coolsp-1.5):
            mode=0;offtime=0.;runtime=0.;cool_integral=0.
        if mode==0 and offtime>=minoff:
            if e>db:
                mode=1;runtime=0.;frac=minfrac;integral=0.
            elif sensed>coolsp+cooldb:
                mode=-1;runtime=0.;frac=.28;cool_integral=0.
        heat=0.;ep=0.;defrosting=False
        if mode==1:
            runtime+=1;offtime=0.
            # Anti-windup PI with a finite compressor slew rate.
            candidate=max(0., min(1./max(ki,.01), integral+e/60))
            if kp*e+ki*candidate<1 or e<0:
                integral=candidate
            target=max(minfrac,min(1.,kp*e+ki*integral))
            frac+=max(-ramp,min(ramp,target-frac))
            cleanp=fan+(maxpower+.005*to-fan)*frac
            if moisture_model and defleft==0:
                rate,drop0,drop1,melt=control[19:23]
                coil=to-drop0-drop1*frac
                vapor=.61078*np.exp(17.27*to/(to+237.3))*rh/100
                sat=.61078*np.exp(21.875*coil/(coil+265.5))
                if coil<0:
                    frost+=rate*frac*max(0.,vapor-sat)
            elif not moisture_model:
                if to<7 and to>-15 and rh>65:
                    tf=np.exp(-max(0.,-to-3)/12)
                    frost+=frate*frac*max(0.,(rh-65)/35)*tf
                elif to>7:
                    frost=max(0.,frost-.007*(to-7))
            if forced_defrost and inp[i,11]>.5:
                defleft=7;frost=0.
            elif (not forced_defrost) and frost>=1 and defleft==0 and (len(control)<=28 or control[28]<.5 or i%5==0):
                defleft=7;frost=0.
            auxgain=thermal[25] if len(thermal)>25 else h0/2.8
            h=max(1.1*auxgain,h0+h1*to)
            if defleft>0:
                ep=12.463;heat=defq;defleft-=1;defrosting=True
            else:
                ep=fan+(cleanp-fan)*(1-fcap*frost)/(1-fcop*frost)
                heat=h*(ep-fan)*(1-fcop*frost)
            # Programmed recovery first uses the compressor; resistance heat
            # becomes available close to the scheduled occupied temperature.
            recovering=(6.5<=hr<6.916667) if we else ((4.5<=hr<5.916667) or (15.5<=hr<16.916667))
            prior=away if not we and hr>12 else night
            auxiliary_error=prior-sensed if recovering else e
            if to<auxlock and auxiliary_error>auxerr and (runtime>auxdelay or auxiliary_error>2.5):
                aux=True
            if to>=auxlock+1 or auxiliary_error<auxoff or mode!=1:
                aux=False
            if aux and not defrosting:
                ep+=auxsize;heat+=auxsize*auxgain
        elif mode==-1:
            runtime+=1;offtime=0.;aux=False;integral=0.
            ec=sensed-coolsp
            candidate=max(0.,min(1./max(ki,.01),cool_integral+ec/60))
            if kp*ec+ki*candidate<1 or ec<0:
                cool_integral=candidate
            target=max(.28,min(1.,kp*ec+ki*cool_integral))
            frac+=max(-ramp,min(ramp,target-frac))
            cool_max=2.832+.0236*(to-25) if sensor_tau>0 else 4.1
            ep=.15+(cool_max-.15)*frac
            humidity_term=thermal[26]*(rh-50)/50 if len(thermal)>26 else 0.
            heat=-max(2.,coolh+cooltc*(to-25)+humidity_term)*(ep-.15)
        else:
            offtime+=1;runtime=0.;aux=False;defleft=0;integral=0.
            if to<10:ep=.045
            if moisture_model:
                frost=max(0.,frost-control[22]*max(0.,to))
            elif to>3:
                frost=max(0.,frost-.002*(to-3))
        qa=g[cl]+pe*base;qm=0.
        for k in range(4):
            qa+=sa[k]*inp[i,3+k];qm+=sm[k]*inp[i,3+k]
        conductance=a+aw*wind
        if len(thermal)>28:
            conductance+=thermal[27]*wind*wind+thermal[28]*np.sqrt(abs(air-to))
        da=conductance*(to-air)+b*(mass-air)+qa+heat
        dm=c*(air-mass)+d*(to-mass)+qm
        air+=da/60;mass+=dm/60
        sensed+=sensor_alpha*(air-sensed)
        if observed is not None:
            sensed=observed[i]
        j=i//5
        result[j,0]+=(base+ep)/5
        result[j,1]=sensed
        result[j,2]+=ep/5
        result[j,3]+=base/5
        result[j,4]+=defrosting/5
        result[j,5]+=aux/5
        result[j,6]=sp
        result[j,7]=mass
        result[j,8]+=frac/5
        result[j,9]=frost
        result[j,10]+=(mode==1)/5
    return result



# Fitted coefficients, stored inline to keep the predictor self-contained.
THERMAL = np.array([0.06698131620479522, 0.0032360747383300653, 1.6169734579149466, 0.21650186975431243, 0.03253313147562138, 0.0, 1.2856711521705495, 0.8354334891982642, 0.09392351568873133, 0.0, 0.80557759069865, 0.0, 0.0648794780851484, 0.1084329851460316, 0.39543220319721883, 0.09768397921638278, 0.3575741968119829, 0.3854160085034258, 0.999980134351608, 25.0, 4.651224457754814, 0.10153026348064574, 0.15, 0.03233635678084747, 6.920689370486129, 1.5999979967897588, -1.572399481882885, 0.0006745608484541375, 0.02], dtype=float)
CONTROL = np.array([0.37014229746738003, 0.38568394020013574, 0.07877318253639053, 7.361546575391557, 4.740572600939561, 0.34353476449564535, 0.025, 0.15, 0.0, 15.061162636221296, 1.0, 25.0, 1.5531085085717338, 0.5358726329645902, 10.0, 0.8750922518638643, 6.732664704581399, -0.08505400306207905, 3.167794747428236, 0.08894374431257494, 4.198659704600968, 4.576403174866022, 0.0028842281185174935, 21.10239979232389, 18.1052555052347, 17.495873175131912, 25.556806706694097, 0.41591348029250974, 1.0], dtype=float)


def predict(weather):
    original_time = weather['time'].copy()
    if len(weather) == 0:
        return pd.DataFrame(columns=['time', 'power_kW', 'indoor_temp_C'])
    parsed = pd.DatetimeIndex(pd.to_datetime(weather['time'], errors='raise'))
    if parsed.hasnans:
        raise ValueError('Weather contains a missing timestamp')
    work = weather.copy()
    work['time'] = parsed
    work = work.sort_values('time').drop_duplicates('time', keep='last')
    times, _, inputs = make_inputs(work, random_loads=False)
    result = simulate(inputs, THERMAL, CONTROL)
    positions = (parsed.asi8-times[0].value)//300_000_000_000
    if np.any((parsed.asi8-times[0].value)%300_000_000_000):
        raise ValueError('Weather timestamps must lie on a five-minute grid')
    power = result[positions, 0]
    temperature = result[positions, 1].copy()
    # Match the instrument's finite precision and small reading noise. Noise is
    # indexed by date, so overlapping forecasts use the same instrument trace.
    for date in parsed.normalize().unique():
        ix = np.flatnonzero(parsed.normalize() == date)
        day_number = int(date.value//86_400_000_000_000)
        rng = np.random.default_rng(day_number*1069+101)
        sensor_noise = rng.normal(0, .06, 288)
        slots = (parsed.hour*12+parsed.minute//5).to_numpy()[ix]
        temperature[ix] += sensor_noise[slots]
    return pd.DataFrame({'time': original_time.to_numpy(),
                         'power_kW': np.round(power, 3),
                         'indoor_temp_C': np.round(temperature, 1)})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('weather_csv')
    parser.add_argument('out_csv')
    args = parser.parse_args()
    weather = pd.read_csv(args.weather_csv)
    predict(weather).to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()
