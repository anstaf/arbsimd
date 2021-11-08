#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_write_cai_breakpoint {

using ::arb::math::exprelr;
using ::arb::math::safeinv;
using ::std::abs;
using ::std::cos;
using ::std::exp;
using ::std::log;
using ::std::max;
using ::std::min;
using ::std::pow;
using ::std::sin;

static constexpr unsigned simd_width_ = 1;
static constexpr unsigned min_align_ = std::max(alignof(arb_value_type), alignof(arb_index_type));

#define PPACK_IFACE_BLOCK \
[[maybe_unused]] auto  _pp_var_width             = pp->width;\
[[maybe_unused]] auto  _pp_var_n_detectors       = pp->n_detectors;\
[[maybe_unused]] auto* _pp_var_vec_ci            = pp->vec_ci;\
[[maybe_unused]] auto* _pp_var_vec_di            = pp->vec_di;\
[[maybe_unused]] auto* _pp_var_vec_t             = pp->vec_t;\
[[maybe_unused]] auto* _pp_var_vec_dt            = pp->vec_dt;\
[[maybe_unused]] auto* _pp_var_vec_v             = pp->vec_v;\
[[maybe_unused]] auto* _pp_var_vec_i             = pp->vec_i;\
[[maybe_unused]] auto* _pp_var_vec_g             = pp->vec_g;\
[[maybe_unused]] auto* _pp_var_temperature_degC  = pp->temperature_degC;\
[[maybe_unused]] auto* _pp_var_diam_um           = pp->diam_um;\
[[maybe_unused]] auto* _pp_var_time_since_spike  = pp->time_since_spike;\
[[maybe_unused]] auto* _pp_var_node_index        = pp->node_index;\
[[maybe_unused]] auto* _pp_var_peer_index        = pp->peer_index;\
[[maybe_unused]] auto* _pp_var_multiplicity      = pp->multiplicity;\
[[maybe_unused]] auto* _pp_var_weight            = pp->weight;\
[[maybe_unused]] auto& _pp_var_events            = pp->events;\
[[maybe_unused]] auto& _pp_var_mechanism_id      = pp->mechanism_id;\
[[maybe_unused]] auto& _pp_var_index_constraints = pp->index_constraints;\
[[maybe_unused]] auto* _pp_var_cai = pp->state_vars[0];\
[[maybe_unused]] auto& _pp_var_ion_ca = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_ca_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
}

static void advance_state(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        _pp_var_cai[i_] =  0.00051999999999999995;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
}

static void write_ions(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[i_];
        arb_value_type cai_shadowed_ = 0;
        cai_shadowed_ = _pp_var_cai[i_];
        _pp_var_ion_ca.internal_concentration[ion_ca_indexi_] = fma(_pp_var_weight[i_], cai_shadowed_, _pp_var_ion_ca.internal_concentration[ion_ca_indexi_]);
    }
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_write_cai_breakpoint
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_write_cai_breakpoint_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_write_cai_breakpoint::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_write_cai_breakpoint::min_align_;
    result.init_mechanism = testing::kernel_write_cai_breakpoint::init;
    result.compute_currents = testing::kernel_write_cai_breakpoint::compute_currents;
    result.apply_events = testing::kernel_write_cai_breakpoint::apply_events;
    result.advance_state = testing::kernel_write_cai_breakpoint::advance_state;
    result.write_ions = testing::kernel_write_cai_breakpoint::write_ions;
    result.post_event = testing::kernel_write_cai_breakpoint::post_event;
    return &result;
  }}

