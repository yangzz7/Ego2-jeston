#include "imu_interval.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void check(bool good) { if(!good) throw std::runtime_error("IMU interval regression"); }
int main() {
    using P=ORB_SLAM3::IMU::Point;
    std::deque<P> q;
    for(int i=0;i<8;++i) q.emplace_back(float(i),0,0,0,0,0,1+i*.005);
    std::vector<P> a,b;
    check(ego2::take_interval(q,1.012,.05,a)==ego2::Interval::ready);
    check(a.size()==4 && std::abs(a.back().a.x()-2.4)<1e-5 && q.front().t==1.012);
    // An intentionally skipped image interval must still deliver every IMU.
    check(ego2::take_interval(q,1.032,.05,b)==ego2::Interval::ready);
    check(b.size()==6 && b.front().t==a.back().t && b.back().t==1.032);
    double sent=-1;
    auto first=ego2::only_new_samples(a,sent);
    auto second=ego2::only_new_samples(b,sent);
    check(first.size()==4 && second.size()==5 && second.front().t>first.back().t);
    for(size_t i=1;i<second.size();++i) check(second[i].t>second[i-1].t);
    check(q.size()==2 && q.back().t>1.032);
    std::vector<P> empty;
    check(ego2::take_interval(q,1.1,.05,empty)==ego2::Interval::waiting && empty.empty());
    q.clear(); q.emplace_back(0,0,0,0,0,0,2.0); q.emplace_back(0,0,0,0,0,0,2.2);
    check(ego2::take_interval(q,2.01,.05,empty)==ego2::Interval::gap);
    std::cout << "IMU boundary, skipped RGB, future bracket and gap checks passed\n";
}
