#ifndef CUDA_SPINAL_PATTERN_GENERATOR_CUH
#define CUDA_SPINAL_PATTERN_GENERATOR_CUH

#include <cuda_runtime.h>
#include <cstddef>

namespace spinalbridge {

struct SensoryAfferentFeedback {
    float hip_angle_rad;
    float knee_angle_rad;
    float load_sensor_N;
};

struct EESPulse {
    float amplitude_mA;
    float frequency_hz;
    float pulse_width_us;
    int spinal_root;
    int gait_phase;
};

class CudaSpinalPatternGenerator {
public:
    static constexpr size_t SPINAL_CHANNELS = 64;

    CudaSpinalPatternGenerator();
    ~CudaSpinalPatternGenerator();

    CudaSpinalPatternGenerator(const CudaSpinalPatternGenerator&) = delete;
    CudaSpinalPatternGenerator& operator=(const CudaSpinalPatternGenerator&) = delete;

    void synthesize_ees_patterns(const float* host_intent_2d, 
                                 const SensoryAfferentFeedback* host_sensory,
                                 float gait_phase_rad,
                                 EESPulse* host_ees_output);

private:
    float* host_mapped_intent_{nullptr};
    float* device_intent_ptr_{nullptr};

    SensoryAfferentFeedback* host_mapped_sensory_{nullptr};
    SensoryAfferentFeedback* device_sensory_ptr_{nullptr};

    EESPulse* host_mapped_ees_{nullptr};
    EESPulse* device_ees_ptr_{nullptr};

    cudaStream_t stream_;
};

} // namespace spinalbridge

#endif
