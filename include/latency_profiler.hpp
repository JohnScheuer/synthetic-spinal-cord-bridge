#ifndef LATENCY_PROFILER_HPP
#define LATENCY_PROFILER_HPP

#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

namespace spinalbridge {

class LatencyProfiler {
public:
    struct Statistics {
        size_t total_samples;
        double min_us;
        double max_us;
        double mean_us;
        double p50_us;
        double p90_us;
        double p99_us;
        double p99_9_us;
        double sla_violations_percent;
        double sla_target_us;
    };

    explicit LatencyProfiler(double sla_target_us = 500.0)
        : sla_target_us_(sla_target_us) {
        samples_.reserve(100000);
    }

    void record_latency(double latency_us) {
        samples_.push_back(latency_us);
    }

    Statistics compute_statistics() {
        if (samples_.empty()) {
            return Statistics{0, 0, 0, 0, 0, 0, 0, 0, 0, sla_target_us_};
        }

        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());

        size_t n = sorted.size();
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        double mean = sum / n;

        size_t sla_count = 0;
        for (double val : sorted) {
            if (val > sla_target_us_) {
                sla_count++;
            }
        }

        auto percentile = [&sorted, n](double p) -> double {
            size_t idx = static_cast<size_t>(std::ceil(p * n)) - 1;
            idx = std::clamp(idx, size_t(0), n - 1);
            return sorted[idx];
        };

        return Statistics{
            n,
            sorted.front(),
            sorted.back(),
            mean,
            percentile(0.50),
            percentile(0.90),
            percentile(0.99),
            percentile(0.999),
            (static_cast<double>(sla_count) / n) * 100.0,
            sla_target_us_
        };
    }

    void print_report(const std::string& title = "SPINAL CORD BRIDGE LATENCY BENCHMARK") {
        Statistics stats = compute_statistics();
        std::cout << "\n======================================================================\n";
        std::cout << " " << title << "\n";
        std::cout << "======================================================================\n";
        std::cout << " Total Closed-Loop Cycles Sampled: " << stats.total_samples << "\n";
        std::cout << " Target Sub-Millisecond SLA    : " << stats.sla_target_us << " us (0.500 ms)\n";
        std::cout << " SLA Violation Rate            : " << std::fixed << std::setprecision(4) 
                  << stats.sla_violations_percent << " %\n";
        std::cout << "----------------------------------------------------------------------\n";
        std::cout << " Minimum Latency                : " << std::setprecision(2) << stats.min_us << " us\n";
        std::cout << " Mean Latency                   : " << std::setprecision(2) << stats.mean_us << " us\n";
        std::cout << " P50 Latency (Median)          : " << std::setprecision(2) << stats.p50_us << " us\n";
        std::cout << " P90 Latency                   : " << std::setprecision(2) << stats.p90_us << " us\n";
        std::cout << " P99 Latency (SLA Target)      : " << std::setprecision(2) << stats.p99_us << " us\n";
        std::cout << " P99.9 Latency                 : " << std::setprecision(2) << stats.p99_9_us << " us\n";
        std::cout << " Maximum Latency               : " << std::setprecision(2) << stats.max_us << " us\n";
        std::cout << "======================================================================\n\n";
    }

private:
    double sla_target_us_;
    std::vector<double> samples_;
};

} // namespace spinalbridge

#endif
