#ifndef PHASE_OSCILLATOR_HPP
#define PHASE_OSCILLATOR_HPP

#include <cmath>
#include <algorithm>

namespace spinalbridge {

// Matsuoka-Kuramoto Adaptive Phase Oscillator for Gait Rhythm Stabilization
class PhaseOscillator {
public:
    PhaseOscillator(float base_frequency_hz = 1.0f) 
        : phase_rad_(0.0f), base_freq_hz_(base_frequency_hz) {}

    // Updates gait phase theta in [0, 2*PI] based on cortical drive and time delta
    float update(float flex_drive, float ext_drive, float dt_sec) {
        float drive_diff = flex_drive - ext_drive;
        
        // Modulate frequency based on supraspinal locomotor intent
        float current_freq = base_freq_hz_ + drive_diff * 0.5f;
        current_freq = std::clamp(current_freq, 0.4f, 2.5f); // 0.4 Hz (slow walk) to 2.5 Hz (fast walk)

        phase_rad_ += 2.0f * M_PI * current_freq * dt_sec;
        if (phase_rad_ >= 2.0f * M_PI) {
            phase_rad_ -= 2.0f * M_PI;
        }

        return phase_rad_;
    }

    float current_phase() const { return phase_rad_; }

private:
    float phase_rad_;
    float base_freq_hz_;
};

} // namespace spinalbridge

#endif
