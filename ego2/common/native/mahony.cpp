// CPU, double precision, one independent state per camera. No camera/GPU/I/O.
#include <array>
#include <deque>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstddef>
using V=std::array<double,3>;using Q=std::array<double,4>;using Row=std::array<double,7>;
static constexpr double NANV=std::numeric_limits<double>::quiet_NaN();
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(V a){return std::sqrt(dot(a,a));}
V add(V a,V b,double k=1){for(int i=0;i<3;i++)a[i]+=k*b[i];return a;}
V scale(V a,double k){for(auto &v:a)v*=k;return a;}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
Q unit(Q q){double n=0;for(auto v:q)n+=v*v;n=std::sqrt(n);if(!std::isfinite(n)||n<1e-12)throw std::runtime_error("invalid quaternion");for(auto &v:q)v/=n;return q;}
Q mul(Q a,Q b){double w=a[0],x=a[1],y=a[2],z=a[3],v=b[0],i=b[1],j=b[2],k=b[3];return {w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v};}
Q conj(Q q){return {q[0],-q[1],-q[2],-q[3]};}
Q inc(V v){double angle=norm(v);if(angle<1e-12)return unit({1,.5*v[0],.5*v[1],.5*v[2]});double k=std::sin(angle/2)/angle;return {std::cos(angle/2),v[0]*k,v[1]*k,v[2]*k};}
Q align(V a){a=scale(a,1/norm(a));if(a[2]<-1+1e-9){int i=0;for(int j=1;j<3;j++)if(std::abs(a[j])<std::abs(a[i]))i=j;V b={0,0,0};b[i]=1;V ax=cross(a,b);ax=scale(ax,1/norm(ax));return {0,ax[0],ax[1],ax[2]};}return unit({1+a[2],a[1],-a[0],0});}
struct State {
 Q q{},ref{},prev{},qic;V bias{},lastg{};bool initialized=false,lastg_valid=false;
 double last=NANV,lastcheck=-INFINITY,weight=0,sparse_until=-INFINITY,recentgap=0;
 double kp,init,maxgap,sparse,gravity,glimit,gstd,astd,blimit,btau;
 long segment=0,valid=0,resets=0,gaps=0;int status=0;
 std::deque<Row> window;std::array<double,6> sum{},sum2{};
 State(const double* qi,const double* s):qic{qi[0],qi[1],qi[2],qi[3]},kp(s[0]),init(s[1]),maxgap(s[2]),sparse(s[3]),gravity(s[4]),glimit(s[5]),gstd(s[6]),astd(s[7]),blimit(s[8]),btau(s[9]){}
 void reset(int reason=3,bool count=true){initialized=false;bias={0,0,0};last=NANV;lastg_valid=false;sparse_until=-INFINITY;recentgap=0;window.clear();sum={};sum2={};lastcheck=-INFINITY;weight=0;status=reason;if(count)resets++;}
 void pop(){auto row=window.front();window.pop_front();for(int j=0;j<6;j++){sum[j]-=row[j+1];sum2[j]-=row[j+1]*row[j+1];}}
 bool stationary(double t,V a,V g,V &am,V &gm){
  if(window.size()==4096)pop();
  window.push_back({t,a[0],a[1],a[2],g[0],g[1],g[2]});
  for(int j=0;j<6;j++){double v=window.back()[j+1];sum[j]+=v;sum2[j]+=v*v;}
  while(window.size()>1&&window[1][0]<=t-init)pop();
  if(t-lastcheck<.1)return false;
  lastcheck=t;
  if(window.size()<100||t-window.front()[0]<init-.001)return false;
  double n=window.size(),ma=0,mg=0;
  for(int j=0;j<3;j++){am[j]=sum[j]/n;gm[j]=sum[j+3]/n;ma=std::max(ma,std::sqrt(std::max(0.,sum2[j]/n-am[j]*am[j])));mg=std::max(mg,std::sqrt(std::max(0.,sum2[j+3]/n-gm[j]*gm[j])));}
  double lim=initialized?blimit:glimit;
  if(initialized&&norm(add(g,bias,-1))>2*lim)return false;
  return std::abs(norm(am)-gravity)<=.08*gravity&&ma<=astd&&mg<=gstd&&norm(add(gm,bias,-1))<=lim;
 }
 bool update(double t,V a,V g,Q &out){
  if(std::isfinite(last)&&t<=last)return false;
  double dt=std::isfinite(last)?t-last:0;
  if(dt>maxgap+1e-9){reset(2);dt=0;}
  bool gap=dt>sparse+1e-9;
  if(gap){gaps++;recentgap=std::max(t<sparse_until?recentgap:0.,dt);sparse_until=t+1;}
  last=t;V am{},gm{};bool rest=stationary(t,a,g,am,gm);
  if(!initialized){
   if(!rest){lastg=g;lastg_valid=true;return false;}
   bias=gm;q=align(am);ref=mul(q,qic);prev={1,0,0,0};initialized=true;segment++;
  }else{
   V rate=(gap&&lastg_valid)?scale(add(lastg,g),.5):g;rate=add(rate,bias,-1);
   Q predicted=unit(mul(q,inc(scale(rate,dt))));
   double w=predicted[0],x=predicted[1],y=predicted[2],z=predicted[3];
   V up={2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
   double n=norm(a);V measured=n>1e-9?scale(a,1/n):up;
   double angle=std::acos(std::clamp(dot(measured,up),-1.,1.));
   double nw=n>1e-9?std::clamp((.20-std::abs(n/gravity-1))/.15,0.,1.):0.;
   double dw=std::clamp((25*M_PI/180-angle)/(15*M_PI/180),0.,1.);
   double target=nw*dw,tau=target<weight?.02:.3;
   weight+=(1-std::exp(-dt/tau))*(target-weight);
   q=unit(mul(predicted,inc(scale(cross(measured,up),kp*weight*std::min(dt,sparse)))));
   if(rest)bias=add(bias,scale(add(gm,bias,-1),1-std::exp(-.1/btau)));
  }
  lastg=g;lastg_valid=true;status=t<sparse_until?4:5;
  out=unit(mul(conj(ref),mul(q,qic)));double d=0;for(int i=0;i<4;i++)d+=out[i]*prev[i];if(d<0)for(auto &v:out)v=-v;
  prev=out;valid++;return true;
 }
 void info(double* o){o[0]=status;o[1]=segment;o[2]=resets;o[3]=gaps;o[4]=recentgap;o[5]=valid;}
};
extern "C" {
 int ego2_mahony_abi(){return 1;}
 void* ego2_mahony_new(const double* q,const double* s){try{return new State(q,s);}catch(...){return nullptr;}}
 void ego2_mahony_delete(void* p){delete static_cast<State*>(p);}
 int ego2_mahony_batch(void* ptr,const double* rows,size_t count,double* out,double* info){
  try {auto &s=*static_cast<State*>(ptr);for(size_t k=0;k<count;k++){
   const double* r=rows+8*k;Q q={NANV,NANV,NANV,NANV};bool finite=true;for(int j=1;j<8;j++)finite&=std::isfinite(r[j]);
   if(!finite)s.reset(3);else s.update(r[1],{r[2],r[3],r[4]},{r[5],r[6],r[7]},q);
   std::copy(q.begin(),q.end(),out+4*k);s.info(info+6*k);
  }return 0;}catch(...){return -1;}
 }
 void ego2_mahony_reset(void* p,int reason,int count){static_cast<State*>(p)->reset(reason,count);}
 void ego2_mahony_info(void* p,double* out){static_cast<State*>(p)->info(out);}
 int ego2_mahony_state(void* p,double* out){auto &s=*static_cast<State*>(p);std::copy(s.q.begin(),s.q.end(),out);std::copy(s.ref.begin(),s.ref.end(),out+4);std::copy(s.bias.begin(),s.bias.end(),out+8);return s.initialized;}
}
