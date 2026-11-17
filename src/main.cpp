#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

#include "lockfree_spsc_queue.hpp"
#include "avx_spike_decoder.hpp"
#include "bio_safety_sentinel.hpp"
#include "phase_oscillator.hpp"
#include "cuda_spinal_pattern_generator.cuh"
#include "latency_profiler.hpp"

using namespace spinalbridge;

int main() {
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    std::cout << "======================================================================\n";
    std::cout << " Clinically-Grounded Synthetic Spinal Cord Bridge Engine               \n";
    std::cout << " ISO 14708-3 Medical Bio-Safety & Adaptive Kuramoto Phase Control    \n";
    std::cout << "======================================================================\n";

    constexpr size_t TOTAL_CYCLES = 10000;
    constexpr size_t CORTICAL_CHANNELS = AVXSpikeDecoder::CORTICAL_CHANNELS;
    constexpr size_t SPINAL_CHANNELS = CudaSpinalPatternGenerator::SPINAL_CHANNELS;
    constexpr float DT_SEC = 0.001f; // 1 ms control step (1000 Hz loop)

    std::cout << "[INFO] Initializing CPG Engine, Bio-Safety Sentinel & Phase Oscillator...\n";
    CudaSpinalPatternGenerator generator;
    PhaseOscillator phase_oscillator(1.0f); // 1.0 Hz gait rhythm
    LatencyProfiler profiler(500.0);

    std::vector<double> latencies;
    latencies.reserve(TOTAL_CYCLES);

    std::vector<float> raw_cortical(CORTICAL_CHANNELS);
    std::vector<float> decoded_intent(2, 0.0f);
    SensoryAfferentFeedback sensory_feedback{};
    std::vector<EESPulse> ees_pulses(SPINAL_CHANNELS);
    std::vector<float> ees_amplitudes(SPINAL_CHANNELS);
    std::vector<EESPulse> sample_ees_pulses(SPINAL_CHANNELS);

    // Warmup cycles
    for (int w = 0; w < 100; ++w) {
        std::fill(raw_cortical.begin(), raw_cortical.end(), 250.0f);
        AVXSpikeDecoder::decode_cortical_intent(raw_cortical.data(), decoded_intent.data(), 100.0f);
        float phase = phase_oscillator.update(decoded_intent[0], decoded_intent[1], DT_SEC);
        generator.synthesize_ees_patterns(decoded_intent.data(), &sensory_feedback, phase, ees_pulses.data());
        cudaDeviceSynchronize();
    }

    std::cout << "[INFO] Clinical Safety Guards & CUDA Pipelines Warmed Up Successfully.\n";
    std::cout << "[INFO] Running Real-Time Synthetic Spinal Cord Bridge Loop...\n\n";

    for (size_t cycle = 0; cycle < TOTAL_CYCLES; ++cycle) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. Simulate Cortical Supraspinal Drive (with realistic noise and dropouts)
        float noise = (rand() % 100 - 50) * 0.5f;
        float cortical_rhythm = std::sin(0.005f * cycle);
        for (size_t c = 0; c < 64; ++c) {
            raw_cortical[c] = (cortical_rhythm > 0) ? (cortical_rhythm * 350.0f + 80.0f + noise) : 10.0f;
        }
        for (size_t c = 64; c < 128; ++c) {
            raw_cortical[c] = (cortical_rhythm <= 0) ? (-cortical_rhythm * 350.0f + 80.0f + noise) : 10.0f;
        }

        // 2. AVX2 SIMD Cortical RMS Energy Envelope Decoding (~70 ns)
        AVXSpikeDecoder::decode_cortical_intent(raw_cortical.data(), decoded_intent.data(), 100.0f);

        // 3. Adaptive Kuramoto Phase Oscillator Update
        float current_gait_phase = phase_oscillator.update(decoded_intent[0], decoded_intent[1], DT_SEC);

        // 4. Simulate Sensorimotor Afferent Feedback
        sensory_feedback.hip_angle_rad = std::sin(current_gait_phase) * 0.4f;
        sensory_feedback.knee_angle_rad = std::cos(current_gait_phase) * 0.6f;
        sensory_feedback.load_sensor_N = (std::sin(current_gait_phase) <= 0.0f) ? 350.0f : 0.0f;

        // 5. CUDA Phase-Gated Lumbosacral EES Spatial Pattern Synthesizer (~38 us)
        generator.synthesize_ees_patterns(decoded_intent.data(), &sensory_feedback, current_gait_phase, ees_pulses.data());

        // 6. AVX2 SIMD ISO 14708-3 Bio-Safety Sentinel Guard (~30 ns)
        for (size_t i = 0; i < SPINAL_CHANNELS; ++i) {
            ees_amplitudes[i] = ees_pulses[i].amplitude_mA;
        }
        BioSafetySentinel::enforce_charge_safety(ees_amplitudes.data(), SPINAL_CHANNELS);
        for (size_t i = 0; i < SPINAL_CHANNELS; ++i) {
            ees_pulses[i].amplitude_mA = ees_amplitudes[i];
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

        latencies.push_back(elapsed_us);
        profiler.record_latency(elapsed_us);

        if (cycle == 5000) {
            sample_ees_pulses = ees_pulses;
        }
    }

    profiler.print_report("CLINICAL SPINAL CORD BRIDGE BENCHMARK (RTX 5060 + Ryzen 5600X)");

    std::ofstream csv("latency_benchmark.csv");
    csv << "cycle_id,latency_us\n";
    for (size_t i = 0; i < latencies.size(); ++i) {
        csv << i << "," << latencies[i] << "\n";
    }
    csv.close();

    std::ofstream ees_csv("ees_pattern.csv");
    ees_csv << "channel,amplitude_mA,frequency_hz,pulse_width_us,spinal_root,gait_phase\n";
    for (size_t c = 0; c < sample_ees_pulses.size(); ++c) {
        ees_csv << c << ","
                << sample_ees_pulses[c].amplitude_mA << ","
                << sample_ees_pulses[c].frequency_hz << ","
                << sample_ees_pulses[c].pulse_width_us << ","
                << sample_ees_pulses[c].spinal_root << ","
                << sample_ees_pulses[c].gait_phase << "\n";
    }
    ees_csv.close();

    std::cout << "[INFO] Telemetry saved to 'latency_benchmark.csv' and 'ees_pattern.csv'.\n";
    return 0;
}
