#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_test5_nonlinear_diff {

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
[[maybe_unused]] auto* _pp_var_c = pp->state_vars[2];\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        _pp_var_a[i_] =  0.20000000000000001;
        _pp_var_b[i_] =  0.29999999999999999;
        _pp_var_c[i_] =  0.5;
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
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type t_16_, t_15_, t_13_, t_12_, t_11_, t_9_, t_7_, t_6_, t_5_, t_4_, t_10_, p_2_, j_8_, j_4_, j_6_, f_2_, j_2_, t_3_, j_5_, t_17_, t_8_, j_1_, t_14_, j_3_, f1, f_0_, j_7_, f_1_, r0, t_2_, f0, j_0_, p_0_, t_1_, r1, p_1_, t_0_;
        p_0_ = _pp_var_a[i_];
        t_0_ = _pp_var_a[i_];
        p_1_ = _pp_var_b[i_];
        t_1_ = _pp_var_b[i_];
        p_2_ = _pp_var_c[i_];
        t_2_ = _pp_var_c[i_];
        f0 =  2.0;
        r0 =  1.0;
        f1 =  3.0;
        r1 =  0.;
        f_0_ = t_0_-(p_0_+( -f0*t_0_*t_1_+r0*t_2_)*dt);
        f_1_ = t_1_-(p_1_+( -f0*t_0_*t_1_-r1*t_1_+(r0+f1)*t_2_)*dt);
        f_2_ = t_2_-(p_2_+(f0*t_0_*t_1_+r1*t_1_-(r0+f1)*t_2_)*dt);
        j_0_ =  1.0- -f0*t_1_*dt;
        j_1_ =  -( -f0*t_0_*dt);
        j_2_ =  -(r0*dt);
        j_3_ =  -( -f0*t_1_*dt);
        j_4_ =  1.0-( -f0*t_0_-r1)*dt;
        j_5_ =  -((r0+f1)*dt);
        j_6_ =  -(f0*t_1_*dt);
        j_7_ =  -((f0*t_0_+r1)*dt);
        j_8_ =  1.0- -(r0+f1)*dt;
        t_3_ = j_8_*j_0_-j_2_*j_6_;
        t_4_ = j_8_*j_1_-j_2_*j_7_;
        t_5_ = j_8_*f_0_-j_2_*f_2_;
        t_6_ = j_8_*j_3_-j_5_*j_6_;
        t_7_ = j_8_*j_4_-j_5_*j_7_;
        t_8_ = j_8_*f_1_-j_5_*f_2_;
        t_9_ = t_7_*t_3_-t_4_*t_6_;
        t_10_ = t_7_*t_5_-t_4_*t_8_;
        t_11_ = t_7_*j_6_-j_7_*t_6_;
        t_12_ = t_7_*j_8_;
        t_13_ = t_7_*f_2_-j_7_*t_8_;
        t_14_ = t_9_*t_7_;
        t_15_ = t_9_*t_8_-t_6_*t_10_;
        t_16_ = t_9_*t_12_;
        t_17_ = t_9_*t_13_-t_11_*t_10_;
        t_0_ = t_0_-t_10_/t_9_;
        t_1_ = t_1_-t_15_/t_14_;
        t_2_ = t_2_-t_17_/t_16_;
        f_0_ = t_0_-(p_0_+( -f0*t_0_*t_1_+r0*t_2_)*dt);
        f_1_ = t_1_-(p_1_+( -f0*t_0_*t_1_-r1*t_1_+(r0+f1)*t_2_)*dt);
        f_2_ = t_2_-(p_2_+(f0*t_0_*t_1_+r1*t_1_-(r0+f1)*t_2_)*dt);
        j_0_ =  1.0- -f0*t_1_*dt;
        j_1_ =  -( -f0*t_0_*dt);
        j_2_ =  -(r0*dt);
        j_3_ =  -( -f0*t_1_*dt);
        j_4_ =  1.0-( -f0*t_0_-r1)*dt;
        j_5_ =  -((r0+f1)*dt);
        j_6_ =  -(f0*t_1_*dt);
        j_7_ =  -((f0*t_0_+r1)*dt);
        j_8_ =  1.0- -(r0+f1)*dt;
        t_3_ = j_8_*j_0_-j_2_*j_6_;
        t_4_ = j_8_*j_1_-j_2_*j_7_;
        t_5_ = j_8_*f_0_-j_2_*f_2_;
        t_6_ = j_8_*j_3_-j_5_*j_6_;
        t_7_ = j_8_*j_4_-j_5_*j_7_;
        t_8_ = j_8_*f_1_-j_5_*f_2_;
        t_9_ = t_7_*t_3_-t_4_*t_6_;
        t_10_ = t_7_*t_5_-t_4_*t_8_;
        t_11_ = t_7_*j_6_-j_7_*t_6_;
        t_12_ = t_7_*j_8_;
        t_13_ = t_7_*f_2_-j_7_*t_8_;
        t_14_ = t_9_*t_7_;
        t_15_ = t_9_*t_8_-t_6_*t_10_;
        t_16_ = t_9_*t_12_;
        t_17_ = t_9_*t_13_-t_11_*t_10_;
        t_0_ = t_0_-t_10_/t_9_;
        t_1_ = t_1_-t_15_/t_14_;
        t_2_ = t_2_-t_17_/t_16_;
        f_0_ = t_0_-(p_0_+( -f0*t_0_*t_1_+r0*t_2_)*dt);
        f_1_ = t_1_-(p_1_+( -f0*t_0_*t_1_-r1*t_1_+(r0+f1)*t_2_)*dt);
        f_2_ = t_2_-(p_2_+(f0*t_0_*t_1_+r1*t_1_-(r0+f1)*t_2_)*dt);
        j_0_ =  1.0- -f0*t_1_*dt;
        j_1_ =  -( -f0*t_0_*dt);
        j_2_ =  -(r0*dt);
        j_3_ =  -( -f0*t_1_*dt);
        j_4_ =  1.0-( -f0*t_0_-r1)*dt;
        j_5_ =  -((r0+f1)*dt);
        j_6_ =  -(f0*t_1_*dt);
        j_7_ =  -((f0*t_0_+r1)*dt);
        j_8_ =  1.0- -(r0+f1)*dt;
        t_3_ = j_8_*j_0_-j_2_*j_6_;
        t_4_ = j_8_*j_1_-j_2_*j_7_;
        t_5_ = j_8_*f_0_-j_2_*f_2_;
        t_6_ = j_8_*j_3_-j_5_*j_6_;
        t_7_ = j_8_*j_4_-j_5_*j_7_;
        t_8_ = j_8_*f_1_-j_5_*f_2_;
        t_9_ = t_7_*t_3_-t_4_*t_6_;
        t_10_ = t_7_*t_5_-t_4_*t_8_;
        t_11_ = t_7_*j_6_-j_7_*t_6_;
        t_12_ = t_7_*j_8_;
        t_13_ = t_7_*f_2_-j_7_*t_8_;
        t_14_ = t_9_*t_7_;
        t_15_ = t_9_*t_8_-t_6_*t_10_;
        t_16_ = t_9_*t_12_;
        t_17_ = t_9_*t_13_-t_11_*t_10_;
        t_0_ = t_0_-t_10_/t_9_;
        t_1_ = t_1_-t_15_/t_14_;
        t_2_ = t_2_-t_17_/t_16_;
        _pp_var_a[i_] = t_0_;
        _pp_var_b[i_] = t_1_;
        _pp_var_c[i_] = t_2_;
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
} // namespace kernel_test5_nonlinear_diff
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test5_nonlinear_diff_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_test5_nonlinear_diff::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_test5_nonlinear_diff::min_align_;
    result.init_mechanism = testing::kernel_test5_nonlinear_diff::init;
    result.compute_currents = testing::kernel_test5_nonlinear_diff::compute_currents;
    result.apply_events = testing::kernel_test5_nonlinear_diff::apply_events;
    result.advance_state = testing::kernel_test5_nonlinear_diff::advance_state;
    result.write_ions = testing::kernel_test5_nonlinear_diff::write_ions;
    result.post_event = testing::kernel_test5_nonlinear_diff::post_event;
    return &result;
  }}

