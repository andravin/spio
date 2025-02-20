#ifndef SPIO_PIPELINE_H_
#define SPIO_PIPELINE_H_

#include "spio/macros.h"

namespace spio
{
    /// @brief A simple pipeline class.
    /// This class is used to manage the state of a pipeline with multiple stages.
    /// Each stage is represented by a bit in an unsigned integer.
    /// The pipeline can be stepped forward, and the active stages can be checked.
    class Pipeline
    {
    public:
        /// @brief Constructor.
        /// @param state the initial state of the pipeline.
        DEVICE Pipeline(unsigned state = 0) : _state(state) {}

        /// @brief Check if a stage is active.
        /// @param stage the bitmask of the stage to check.
        /// @return True if the stage is active, false otherwise.
        DEVICE bool active(unsigned stage) const { return (stage & _state) != 0; }

        /// @brief Check if two stages are active.
        /// @param stage_1 the bitmask of the first stage to check.
        /// @param stage_2 the bitmask of the second stage to check.
        /// @return True if both stage_1 and stage_2 are active, false otherwise.
        DEVICE bool active(unsigned stage_1, unsigned stage_2) const
        {
            return ((stage_1 | stage_2) & _state) == (stage_1 | stage_2);
        }

        /// @brief Step the pipeline.
        /// All pipeline stages are shifted one bit to the left. Logically this
        /// shifts a task to the next stage in the pipeline.
        /// @param active Set to true to to activate the first pipeline stage.
        DEVICE void step(bool active)
        {
            _state <<= 1;
            _state |= (active ? 1 : 0);
        }

    private:
        unsigned _state;
    };
} // namespace spio

#endif // SPIO_PIPELINE_H_
