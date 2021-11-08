#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_test1_kin_diff {

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
[[maybe_unused]] auto* _pp_var_a = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_b = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_x = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_y = pp->state_vars[3];\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        _pp_var_a[i_] =  0.20000000000000001;
        _pp_var_b[i_] =  1.0-_pp_var_a[i_];
        _pp_var_x[i_] =  0.59999999999999998;
        _pp_var_y[i_] =  1.0-_pp_var_x[i_];
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 4; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type t_7_, t_5_, t_3_, t_2_, t_0_, a_7_, t_6_, a_6_, a_5_, a_4_, t_4_, alpha, a_3_, a_2_, a_1_, delta, t_1_, beta, a_0_, gamma;
        alpha =  2.0;
        beta =  0.59999999999999998;
        gamma =  3.0;
        delta =  0.69999999999999996;
        a_0_ =  1.0- -1.0*alpha*dt;
        a_1_ =  -( -1.0* -beta*dt);
        a_2_ =  -(alpha*dt);
        a_3_ =  1.0- -beta*dt;
        a_4_ =  1.0- -1.0*gamma*dt;
        a_5_ =  -( -1.0* -delta*dt);
        a_6_ =  -(gamma*dt);
        a_7_ =  1.0- -delta*dt;
        t_0_ = a_7_*a_4_-a_5_*a_6_;
        t_1_ = a_7_*_pp_var_x[i_]-a_5_*_pp_var_y[i_];
        t_2_ = t_0_*a_7_;
        t_3_ = t_0_*_pp_var_y[i_]-a_6_*t_1_;
        t_4_ = a_3_*a_0_-a_1_*a_2_;
        t_5_ = a_3_*_pp_var_a[i_]-a_1_*_pp_var_b[i_];
        t_6_ = t_4_*a_3_;
        t_7_ = t_4_*_pp_var_b[i_]-a_2_*t_5_;
        _pp_var_a[i_] = t_5_/t_4_;
        _pp_var_b[i_] = t_7_/t_6_;
        _pp_var_x[i_] = t_1_/t_0_;
        _pp_var_y[i_] = t_3_/t_2_;
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
} // namespace kernel_test1_kin_diff
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test1_kin_diff_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_test1_kin_diff::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_test1_kin_diff::min_align_;
    result.init_mechanism = testing::kernel_test1_kin_diff::init;
    result.compute_currents = testing::kernel_test1_kin_diff::compute_currents;
    result.apply_events = testing::kernel_test1_kin_diff::apply_events;
    result.advance_state = testing::kernel_test1_kin_diff::advance_state;
    result.write_ions = testing::kernel_test1_kin_diff::write_ions;
    result.post_event = testing::kernel_test1_kin_diff::post_event;
    return &result;
  }}

