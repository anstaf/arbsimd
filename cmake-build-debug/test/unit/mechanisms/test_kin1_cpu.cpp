#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_test_kin1 {

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
[[maybe_unused]] auto _pp_var_tau = pp->globals[0];\
[[maybe_unused]] auto* _pp_var_a = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_b = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_v = pp->state_vars[2];\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        _pp_var_a[i_] =  0.01;
        _pp_var_b[i_] =  0.;
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 2; ++ix) {
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
        arb_value_type t_3_, t_0_, a_3_, t_2_, a_2_, a_1_, t_1_, a_0_;
        a_0_ =  1.0- -1.0*( 0.66666666666666663/_pp_var_tau)*dt;
        a_1_ =  -( -1.0* -( 0.33333333333333331/_pp_var_tau)*dt);
        a_2_ =  -( 0.66666666666666663/_pp_var_tau*dt);
        a_3_ =  1.0- -( 0.33333333333333331/_pp_var_tau)*dt;
        t_0_ = a_3_*a_0_-a_1_*a_2_;
        t_1_ = a_3_*_pp_var_a[i_]-a_1_*_pp_var_b[i_];
        t_2_ = t_0_*a_3_;
        t_3_ = t_0_*_pp_var_b[i_]-a_2_*t_1_;
        _pp_var_a[i_] = t_1_/t_0_;
        _pp_var_b[i_] = t_3_/t_2_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type current_ = 0;
        arb_value_type il = 0;
        il = _pp_var_a[i_];
        current_ = il;
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_test_kin1
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test_kin1_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_test_kin1::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_test_kin1::min_align_;
    result.init_mechanism = testing::kernel_test_kin1::init;
    result.compute_currents = testing::kernel_test_kin1::compute_currents;
    result.apply_events = testing::kernel_test_kin1::apply_events;
    result.advance_state = testing::kernel_test_kin1::advance_state;
    result.write_ions = testing::kernel_test_kin1::write_ions;
    result.post_event = testing::kernel_test_kin1::post_event;
    return &result;
  }}

