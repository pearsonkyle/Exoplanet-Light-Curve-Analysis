 /*

 CITATIONS:
    These routines are based on the work in Mandel & Agol (2002),
    so please cite this paper if you make use of these routines
    in your research.

    http://adsabs.harvard.edu/abs/2002ApJ...580L.171M

    gcc -Ofast -shared -o lib_transit.so -fPIC MandelTransit.c
 */

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#define pi 3.14159265358979311600
#define invPi 1./3.1415926535897931160
#define inv3 1./3
#define inv9 1./9
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

void occultquad(double *t, double p, double ar, double P, double i, double gamma1, double gamma2, double e, double longPericenter, double tmid, double n, double *F);
void phasecurve(double *t, double *C, double erprs, double rprs, double ars, double P, double inc, double gamma1, double gamma2, double e, double longPericenter, double tmid, double n, double *F);

// Elliptic integral arproximations
double E(double k);
double PI(double n, double k);
double K(double k);

double PI(double n, double k)
{
	// Computes the complete elliptical integral of the third kind using the algorithm of Bulirsch (1965):
	// Translation of Eric's "ellpic_bulirsch"
	double m0, kc, c, p, d, e, f, dpi, quantity, g;
	int continueLoop;

	dpi = pi;

	kc = sqrt(1.0-k*k);
	p = n+1.0;
	m0 = 1.0;
	c = 1.0;
	p = sqrt(p);
	d = 1.0/p;
	e = kc;

	continueLoop = 1;
	while (continueLoop == 1) {
		f = c;
		c = d/p+f;
		g = e/p;
		d = (f*g+d)*2.0;
		p = g + p;
		g = m0;
		m0 = kc + m0;
		quantity = fabs(1.0-kc/g);
		if (quantity > 1.0e-13) {
			kc = 2.0*sqrt(e);
			e = kc*m0;
		} 
        else { continueLoop = 0; 
        }
	}
	return 0.5*dpi*(c*m0+d)/(m0*(m0+p));
}

double K(double k)
{
	// Translation of Eric's "ellk"
	// Computes polynomial arproximation for the complete elliptic integral
	// of the first kind (Hasting's arproximation):

	double m1, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, ek1, ek2;
	m1 = 1.0-k*k;
	if (k*k > 1) {
        printf("WARNING: k*k > 1, elliptic integral of first kind, K, will return nan");
    }
	a0 = 1.386294361120;
	a1 = 0.096663442590;
	a2 = 0.035900923830;
	a3 = 0.037425637130;
	a4 = 0.014511962120;
	b0 = 0.50;
	b1 = 0.124985935970;
	b2 = 0.068802485760;
	b3 = 0.033283553460;
	b4 = 0.004417870120;
	ek1 = a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)));
	ek2 = (b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*log(m1);
	return ek1-ek2;
}

double E(double k)
{
	// Translation of Eric's "ellec"
	// Computes polynomial arproximation for the complete elliptic integral
	// of the second kind (Hasting's arproximation):
	double m1, a1, a2, a3, a4, b1, b2, b3, b4, ee1, ee2;
	m1 = 1.0-k*k;
	a1=0.44325141463;
	a2=0.06260601220;
	a3=0.04757383546;
	a4=0.01736506451;
	b1=0.24998368310;
	b2=0.09200180037;
	b3=0.04069697526;
	b4=0.00526449639;
	ee1 = 1.0+m1*(a1+m1*(a2+m1*(a3+m1*a4)));
	ee2 = m1*(b1+m1*(b2+m1*(b3+m1*b4)))*log(1.0/m1);
	return ee1+ee2;
}

double heaviside(double x)
{
	double result;
	if (x <= 0) {
		result = 0;
	} 
    else if (x > 0) {
		result = 1;
	} 
	return result;
}

double lam1(double p, double z, double a, double b, double k, double q)
{
	double inva = 1./a;
	double lam1, en;
	en=inva-1.0; 
	lam1 = (((1.0-b)*(2.0*b+a-3.0)-3.0*q*(b-2.0))*K(k)+4.0*p*z*(z*z+7.0*p*p-4.0)*E(k)-3.0*q*inva*PI(en,k))*inv9*invPi/sqrt(p*z); // Eric Agol's code inspired
	return lam1;
}

double lam2(double p, double z, double a, double b, double k, double q)
{
	double inva = 1./a;
	double lam2;
	double invk = 2.0*sqrt(p*z/(1.0-a)); 
	double en=b*inva-1.0;					
	lam2 = 2.0*inv9*invPi/sqrt(1-a) * ( (1-5*z*z+p*p+q*q)*K(invk) + (1-a)*(z*z+7*p*p-4)*E(invk)-3*q*inva*PI(en,invk) );
	return lam2;
}

double lam3(double p, double k0, double k1)
{
	double lam3 = inv3 + 16*p*inv9*invPi*(2*p*p-1)*E(0.5/p) - (1-4*p*p)*(3-8*p*p)*inv9*invPi/p*K(0.5/p);
	return lam3;
}

double lam4(double p)
{
	double lam4 = inv3 + 2.0*inv9*invPi*(4*(2*p*p-1)*E(2*p)+(1-4*p*p)*K(2*p));
	return lam4;
}

double lam5(double p)
{
	double lam5 = 2.0*inv3*invPi*acos(1-2*p) - 4.0*inv9*invPi*(3+2*p-8*p*p)*sqrt(p*(1-p))-2.0*inv3*heaviside(p-0.5);
	return lam5;
}

double lam6(double p)
{
	double lam6 = -2.0*inv3*sqrt((1-p*p)*(1-p*p)*(1-p*p));
	return lam6;
}

double eta2(double p, double z)
{
	double eta2 = p*p*0.5*(p*p+2.0*z*z);
	return eta2;
}

double eta1(double p, double z, double a, double b, double k1, double k0)
{
	double eta1;
	eta1 = 0.5*invPi*(k1+2.0*eta2(p,z)*k0-0.25*(1.0+5.0*p*p+z*z)*sqrt((1.0-a)*(b-1.0)));
	return eta1;

}

double ekepler(double m, double e)
{
	double ekep,eps,pi2,ms,d3,e0,f0,f1,f2,f3,d1,d2;
	eps = 1.0E-10;
	pi2 = 2.0*pi;
	ms = fmod(m,pi2);
	d3 = 1.0E10;
	e0 = ms+e*0.85*sin(ms)/fabs(sin(ms));
	while (fabs(d3) > eps)
	{
		f3 = e*cos(e0);
		f2 = e*sin(e0);
		f1 = 1.0-f3;
		f0 = e0-ms-f2;
		d1 = -1*f0/f1;
		d2 = -1*f0/(f1+0.5*d1*f2);
		d3 = -1*f0/(f1+d2*0.5*(f2+d2*f3/3.0));
		e0 = e0+d3;
	}
	ekep = e0 + m - ms;
	return ekep;
}

double kepler(double m, double e)
{
	double i = 0, f, ekep;
	if (e != 0.0) {
		ekep = ekepler(m,e);
		f = 2.0*atan(sqrt((1.0+e)/(1.0-e))*tan(0.50*ekep));
	} 
    else {
		f = m;
	}
	if (m == 0.0) {
        f=0.0;
    }
	return f;
}

double kepler_opt(double m, double eplusoverminus, double e)
{
	double i = 0, f, ekep;
	if (e != 0.0) {
		ekep = ekepler(m,e);
		f = 2.0*atan(eplusoverminus*tan(0.50*ekep));
	} else {
		f = m;
	}
	if (m == 0.0) {
        f=0.0;
    }
	return f;
}

double occultuni(double z, double w)
{
	double xt, kar1, kar0, lambdae, dblcond, dblcondA, dblcondB, muo1;
	if (fabs(w-0.5) < 1.0E-3) {
		w = 0.5;
	}
	if (z > 1.0+w) {
		muo1 = 1.0;
	}
	if (z > fabs(1.0-w) &&  z <= 1.0+w) {
		xt = (1.0-w*w+z*z)/2.0/z;

		if (xt < 1.0) {
			dblcondA = 1;
		} 
        else {
			dblcondA = 0;
		}
		if (xt >= 1.0) {
			dblcondB = 1;
		} 
        else {
			dblcondB = 0;
		}

		kar1 = acos(xt*dblcondA+1.0*dblcondB);
		xt=(w*w+z*z-1.0)/2.0/w/z;

		kar0 = acos(xt*dblcondA+1.0*dblcondB);
		lambdae=w*w*kar0+kar1;
		xt = 4.0*z*z-(1.0+z*z-w*w)*(1.0+z*z-w*w);

		if (xt >= 0.0) {
			dblcond = 1;
		} 
        else {
			dblcond = 0;
		}

		lambdae = (lambdae-0.5*sqrt(xt*dblcond))/pi;
		muo1 = 1.0-lambdae;
	}

	if (z <= 1.0-w) {
		muo1 = 1.0-w*w ;
	}
	return muo1;
}


void phasecurve(double *t, double *C, double erprs, double rprs, double ars, double P, double inc, double gamma1, double gamma2, double e, double longPericenter, double tmid, double n, double *F)
{
	// transit 
	occultquad(t, rprs, ars, P, inc, gamma1, gamma2, e, longPericenter, tmid, n, F);

	// eclipse
	// https://arxiv.org/pdf/1001.2010.pdf eq 33
	double *eclipse = (double *) malloc(sizeof(double)*(int)n);
	double tme = tmid + P*0.5*(1+e*(4*invPi)*cos(longPericenter*pi/180.));
	occultquad(t, erprs, ars, P, inc, 0, 0, e, longPericenter+180, tme, n, eclipse);

	// offset so that mid-eclipse is 1
	C[0] = -1*((C[1]*cos(2*pi*tme/P) + C[2]*sin(2*pi*tme/P) + C[3]*cos(4*pi*tme/P) + C[4]*sin(4*pi*tme/P))-1*erprs*erprs);

	// phase curve
	for (int i=0; i<(int)n; i++)
	{
		F[i] *= F[i] * (1 + C[0] + C[1]*cos(2*pi*t[i]/P) + C[2]*sin(2*pi*t[i]/P) + C[3]*cos(4*pi*t[i]/P) + C[4]*sin(4*pi*t[i]/P));
		F[i] *= eclipse[i];

		// adjust ingress to mid-eclipse
		if (floor(eclipse[i]+erprs*erprs-1e-6)==0)
		{
			F[i] = 1;
		}
	
		// adjust mid-eclipse to egress
		if (floor(eclipse[i])==0)
		{
			F[i] = max(F[i],1);
		}
	}
	free(eclipse);
}


void occultquad(double *t, double p, double ar, double P, double i, double gamma1, double gamma2, double e, double longPericenter, double tmid, double n, double *F)
{
	double tmidoverP, tmidf;
	double *Z, *phi, *new_phi; int ii;
	int Npoints = (int)n;

    Z = (double *) malloc(sizeof(double)*Npoints);
    phi = (double *) malloc(sizeof(double)*Npoints);

	// optimization parameters
	double invP = 1./P;
	double sqrtee = sqrt(1-e*e);
	double inv180 = 1./180.;
	double epoverm = sqrt((1.0+e)/(1.0-e));

	// phase + modification for time series longer than 1 period
	for (ii=0; ii<Npoints; ii++) 
	{
        phi[ii] = (t[ii]-tmid)*invP;
        phi[ii] = fmod(phi[ii],1);
        if (phi[ii] > 0.5) phi[ii]-=1; 
	}

	double ti;
	double omega;
	for (ii=0; ii<Npoints; ii++)
	{
		ti = t[ii];
		double f1, e1,tp, m, f, radius;
		f1 = 1.50*pi-longPericenter*pi*inv180;
		e1 = e;

		// looping error perhaps comes from somewhere in here
		tp = tmid+P*sqrtee*0.5*invPi*(e1*sin(f1)/(1.0+e1*cos(f1))-2.0/sqrtee*atan( (sqrtee*tan(0.5*f1))/(1.0+e1) ));
		m = 2.0*pi*invP*(ti-tp);
		f = kepler_opt(m,epoverm,e1);
		radius = ar*(1.0 - e1*e1)/(1.0 + e1*cos(f));
		Z[ii] = radius*sqrt(1.0-(sin(i*pi*inv180)*sin(longPericenter*pi*inv180+f))*(sin(i*pi*inv180)*sin(longPericenter*pi*inv180+f))); //Eric Agol's code inspired		
	}

	omega=1.0-gamma1*inv3-gamma2/6.0;

	int j = 0;
	double z, a, b, k, q, k1, k0, lam_e, F0;
	double invz, invp, invar,p2,z2;
	p2 = p*p;
	invp = 1./p;
	invar = 1./ar;

	for (j=0;j<Npoints;j++)
	{
		z = fabs(Z[j]);
		invz = 1./z;
		z2 = z*z;
	    a = (z-p)*(z-p);
	    b = (z+p)*(z+p);
	    k = sqrt((1.0-a)*0.25*invz*invp);
	    q = p2-z2;
	    k1=acos((1-p2+z2)*0.5*invz);
	    k0=acos((p2+z2-1)*0.5*invp*invz);

	    // Evaluate lambda_e, MA2002 eq. (1)
	    if (1+p<z || fabs(phi[j])>(p+1)*invar*0.5*invPi) {
	        lam_e = 0;
	    } else if (fabs(1-p)<z && z<=1+p) {
	        lam_e = 1*invPi*(p2*k0+k1-0.5*sqrt(4*z2-(1+z2-p2)*(1+z2-p2)));
	    } else if (z<=1-p && z>p-1) {
	        lam_e = p2;
	    } else if (z<=p-1) {
	        lam_e = 1;
	    }
	    double lam_d, eta_d;
	    // Evaluate lambda_d and eta_d from MA2002 Table (1)
	    if (z>=(1+p) || p==0 || fabs(phi[j])>(p+1.0)*invar*0.5*invPi) { // Case 1
			lam_d = 0.0;
	        eta_d = 0.0;
	      } else if (p<0.5 && z>p && z<1-p) {// Case 3	-- switch order since most time should be spent in case 3
			lam_d = lam2(p,z,a,b,k,q);
			eta_d = eta2(p,z);
		  } else if (z>=fabs(1.0-p) && z<(1+p)) { // Case 2
	        lam_d = lam1(p,z,a,b,k,q);
	        eta_d = eta1(p,z,a,b,k1,k0);
	      } else if (p<0.5 && z==1-p) {// Case 4
	        lam_d = lam5(p);
	        eta_d = eta2(p,z);
	      } else if (p<0.5 && z==p) { // Case 5
	        lam_d = lam4(p);
	        eta_d = eta2(p,z);
	      } else if (p==0.5 && z==0.5) { // Case 6
	        lam_d = inv3-4.0*invPi*inv9;
	        eta_d = 3.0/32.0;
	      } else if (p>0.5 && z==p) { // Case 7
	        lam_d = lam3(p, k0, k1);
	        eta_d = eta1(p,z,a,b,k1,k0);
	      } else if (p>0.5 && z>=fabs(1.0-p) && z<p) { // Case 8
	        lam_d = lam1(p,z,a,b,k,q);
	        eta_d = eta1(p,z,a,b,k1,k0);
	      } else if (p<1 && z>0 && z<=0.5-fabs(p-0.5)) {// Case 9
	        lam_d = lam2(p,z,a,b,k,q);
	        eta_d = eta2(p,z);
	      } else if (p<1 && z==0) {  // Case 10
	        lam_d = lam6(p);
	        eta_d=eta2(p,z);
	      } else if (p>1 && z<=p-1) {  // Case 11
	    	lam_d = 0.0;
	        eta_d = 0.5;
	    }

		F[j] = 1.0-( (1.0-gamma1-2.0*gamma2)*lam_e+(gamma1+2.0*gamma2)*(lam_d+2.0/3.0*heaviside(p-z))+gamma2*eta_d)/omega; // Eric Agol's code inspired
		if (z>=(1+p) || p==0 || fabs(phi[j])>(p+1.0)*invar*0.5*invPi) { // Case 1
			F[j] = 1; // F[j] != 1 when planet is not in sight
		}
	}
    free(Z);
    free(phi);
}