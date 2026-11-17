#include "cuda_spinal_pattern_generator.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace spinalbridge {

// CUDA Kernel: Continuous Phase-Gated Spatial Activation Profile across L2-S2 Spinal Roots
__global__ void cpg_phase_gated_ees_kernel(const float* __restrict__ intent_2d,
                                           const SensoryAfferentFeedback* __restrict__ sensory,
                                           float gait_phase_rad,
                                           EESPulse* __restrict__ ees_out,
                                           int num_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_channels) {
        float flex_drive = intent_2d[0];
        float ext_drive  = intent_2d[1];
        float ground_force = sensory->load_sensor_N;

        int root = idx / 11; // 64 channels distributed across 6 lumbosacral segments (L2, L3, L4, L5, S1, S2)
        root = min(root, 5);

        // Smooth phase progression:
        // Phase 0..PI: Swing Phase (Flexion / Foot Lift)
        // Phase PI..2*PI: Stance Phase (Extension / Weight Acceptance / Push-off)
        float sin_phase = sinf(gait_phase_rad);
        float cos_phase = cosf(gait_phase_rad);

        float amp_mA = 0.0f;
        float freq_hz = 40.0f;
        int phase_id = (sin_phase > 0.0f) ? 0 : 2;

        switch (root) {
            case 0: // L2 (Hip Flexion - Iliopsoas)
            case 1: // L3 (Knee Flexion)
                if (sin_phase > 0.0f) { // Early/Mid Swing
                    amp_mA = (flex_drive * 7.0f + 1.5f) * sin_phase;
                    freq_hz = 60.0f;
                }
                break;
            case 2: // L4 (Knee Extension - Quadriceps)
                if (sin_phase <= 0.0f && cos_phase < 0.0f) { // Early Stance Acceptance
                    amp_mA = (ext_drive * 8.0f + ground_force * 0.008f) * (-sin_phase);
                    freq_hz = 50.0f;
                }
                break;
            case 3: // L5 (Ankle Dorsiflexion - Tibialis Anterior)
                if (sin_phase > 0.2f) { // Full Swing Clearance
                    amp_mA = (flex_drive * 6.0f) * sin_phase;
                    freq_hz = 80.0f;
                }
                break;
            case 4: // S1 (Plantarflexion - Gastrocnemius / Push-off)
            case 5: // S2 (Proprioceptive Reflex Stabilization)
                if (sin_phase <= 0.0f && cos_phase >= 0.0f) { // Late Stance Push-Off
                    amp_mA = (ext_drive * 8.5f) * (-sin_phase);
                    freq_hz = 45.0f;
                }
                break;
        }

        EESPulse pulse;
        pulse.amplitude_mA   = amp_mA;
        pulse.frequency_hz   = freq_hz;
        pulse.pulse_width_us  = 350.0f;
        pulse.spinal_root    = root;
        pulse.gait_phase      = phase_id;

        ees_out[idx] = pulse;
    }
}

CudaSpinalPatternGenerator::CudaSpinalPatternGenerator() {
    cudaError_t err = cudaHostAlloc((void**)&host_mapped_intent_, 2 * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostAlloc intent failed");
    cudaHostGetDevicePointer((void**)&device_intent_ptr_, host_mapped_intent_, 0);

    err = cudaHostAlloc((void**)&host_mapped_sensory_, sizeof(SensoryAfferentFeedback), cudaHostAllocMapped);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostAlloc sensory failed");
    cudaHostGetDevicePointer((void**)&device_sensory_ptr_, host_mapped_sensory_, 0);

    err = cudaHostAlloc((void**)&host_mapped_ees_, SPINAL_CHANNELS * sizeof(EESPulse), cudaHostAllocMapped);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostAlloc ees failed");
    cudaHostGetDevicePointer((void**)&device_ees_ptr_, host_mapped_ees_, 0);

    cudaStreamCreate(&stream_);
}

CudaSpinalPatternGenerator::~CudaSpinalPatternGenerator() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);

    if (host_mapped_intent_) cudaFreeHost(host_mapped_intent_);
    if (host_mapped_sensory_) cudaFreeHost(host_mapped_sensory_);
    if (host_mapped_ees_) cudaFreeHost(host_mapped_ees_);
}

void CudaSpinalPatternGenerator::synthesize_ees_patterns(const float* host_intent_2d, 
                                                         const SensoryAfferentFeedback* host_sensory,
                                                         float gait_phase_rad,
                                                         EESPulse* host_ees_output) {
    std::memcpy(host_mapped_intent_, host_intent_2d, 2 * sizeof(float));
    std::memcpy(host_mapped_sensory_, host_sensory, sizeof(SensoryAfferentFeedback));

    int threads = 32;
    int blocks = (SPINAL_CHANNELS + threads - 1) / threads;

    cpg_phase_gated_ees_kernel<<<blocks, threads, 0, stream_>>>(
        device_intent_ptr_,
        device_sensory_ptr_,
        gait_phase_rad,
        device_ees_ptr_,
        SPINAL_CHANNELS
    );

    cudaStreamSynchronize(stream_);

    std::memcpy(host_ees_output, host_mapped_ees_, SPINAL_CHANNELS * sizeof(EESPulse));
}

} // namespace spinalbridge
