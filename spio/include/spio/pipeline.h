#ifndef SPIO_PIPELINE_H_
#define SPIO_PIPELINE_H_

#include "spio/macros.h"

namespace spio
{
    /// Manages the state of a pipeline with multiple stages.
    ///
    /// Each stage is represented by a bit in an unsigned integer. The pipeline
    /// can be stepped forward, and active stages can be checked.
    class Pipeline
    {
    public:
        /// Constructs a pipeline with an initial state.
        ///
        /// Parameters:
        ///   state   Initial state bitmask (default 0).
        DEVICE Pipeline(unsigned state = 0) : _state(state) {}

        /// Checks if a stage is active.
        ///
        /// Parameters:
        ///   stage   Bitmask of the stage to check.
        ///
        /// Returns:
        ///   True if the stage is active.
        DEVICE bool active(unsigned stage) const { return (stage & _state) != 0; }

        /// Checks if two stages are both active.
        ///
        /// Parameters:
        ///   stage_1   Bitmask of the first stage.
        ///   stage_2   Bitmask of the second stage.
        ///
        /// Returns:
        ///   True if both stages are active.
        DEVICE bool active(unsigned stage_1, unsigned stage_2) const
        {
            return ((stage_1 | stage_2) & _state) == (stage_1 | stage_2);
        }

        /// Steps the pipeline forward by one stage.
        ///
        /// Shifts all stages one bit to the left, moving tasks to the next stage.
        ///
        /// Parameters:
        ///   active   If true, activates the first pipeline stage.
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
