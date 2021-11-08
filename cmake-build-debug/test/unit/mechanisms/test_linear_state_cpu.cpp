#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_test_linear_state {

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
[[maybe_unused]] auto _pp_var_a4 = pp->globals[0];\
[[maybe_unused]] auto* _pp_var_s = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_d = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_a0 = pp->state_vars[3];\
[[maybe_unused]] auto* _pp_var_a1 = pp->state_vars[4];\
[[maybe_unused]] auto* _pp_var_a2 = pp->state_vars[5];\
[[maybe_unused]] auto* _pp_var_a3 = pp->state_vars[6];\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        _pp_var_a0[i_] =  2.5;
        _pp_var_a1[i_] =  0.5;
        _pp_var_a2[i_] =  3.0;
        _pp_var_a3[i_] =  2.2999999999999998;
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 3; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        arb_value_type t_11_, t_8_, t_7_, t_6_, t_5_, t_1_, l_9_, t_4_, l_7_, l_8_, t_3_, l_6_, t_10_, l_5_, l_4_, t_9_, l_2_, l_1_, t_0_, l_3_, t_2_, l_0_;
        l_0_ = _pp_var_a4-_pp_var_a3[i_];
        l_1_ =  -_pp_var_a2[i_];
        l_2_ =  0.;
        l_3_ = _pp_var_a0[i_]+_pp_var_a1[i_];
        l_4_ =  -( -_pp_var_a1[i_]+_pp_var_a0[i_]);
        l_5_ =  0.;
        l_6_ =  1.0;
        l_7_ =  1.0;
        l_8_ =  1.0;
        l_9_ =  1.0;
        t_0_ =  -(l_1_*l_6_);
        t_1_ = l_8_*l_0_-l_1_*l_7_;
        t_2_ = l_8_*l_2_-l_1_*l_9_;
        t_3_ = l_4_*t_0_-t_1_*l_3_;
        t_4_ = l_4_*t_2_-t_1_*l_5_;
        t_5_ = l_4_*l_6_-l_7_*l_3_;
        t_6_ = l_4_*l_8_;
        t_7_ = l_4_*l_9_-l_7_*l_5_;
        t_8_ = t_3_*l_4_;
        t_9_ = t_3_*l_5_-l_3_*t_4_;
        t_10_ = t_3_*t_6_;
        t_11_ = t_3_*t_7_-t_5_*t_4_;
        _pp_var_s[i_] = t_4_/t_3_;
        _pp_var_d[i_] = t_9_/t_8_;
        _pp_var_h[i_] = t_11_/t_10_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_test_linear_state
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test_linear_state_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_test_linear_state::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_test_linear_state::min_align_;
    result.init_mechanism = testing::kernel_test_linear_state::init;
    result.compute_currents = testing::kernel_test_linear_state::compute_currents;
    result.apply_events = testing::kernel_test_linear_state::apply_events;
    result.advance_state = testing::kernel_test_linear_state::advance_state;
    result.write_ions = testing::kernel_test_linear_state::write_ions;
    result.post_event = testing::kernel_test_linear_state::post_event;
    return &result;
  }}

