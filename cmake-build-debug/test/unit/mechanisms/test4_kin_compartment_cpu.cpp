#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_test4_kin_compartment {

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
[[maybe_unused]] auto _pp_var_x = pp->globals[0];\
[[maybe_unused]] auto _pp_var_y = pp->globals[1];\
[[maybe_unused]] auto _pp_var_z = pp->globals[2];\
[[maybe_unused]] auto _pp_var_w = pp->globals[3];\
[[maybe_unused]] auto _pp_var_s0 = pp->globals[4];\
[[maybe_unused]] auto _pp_var_s1 = pp->globals[5];\
[[maybe_unused]] auto* _pp_var_A = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_B = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_C = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_d = pp->state_vars[3];\
[[maybe_unused]] auto* _pp_var_e = pp->state_vars[4];\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        _pp_var_A[i_] =  4.5;
        _pp_var_B[i_] =  6.5999999999999996;
        _pp_var_C[i_] =  0.28000000000000003;
        _pp_var_d[i_] =  2.0;
        _pp_var_e[i_] =  0.;
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 5; ++ix) {
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
        arb_value_type t_38_, t_36_, t_35_, t_34_, t_33_, t_31_, t_29_, t_26_, t_25_, t_24_, t_23_, t_21_, t_19_, t_16_, t_15_, t_13_, t_28_, t_11_, j_10_, t_12_, t_9_, t_18_, t_7_, t_6_, t_5_, j_16_, t_3_, j_15_, f_0_, f_4_, p_2_, j_9_, t_2_, f_2_, t_17_, j_7_, j_8_, j_12_, f_1_, t_30_, j_2_, p_1_, t_27_, s_3_, s_2_, s_0_, t_32_, t_14_, j_11_, j_3_, j_13_, t_20_, j_0_, p_0_, p_4_, j_5_, t_8_, j_1_, t_4_, j_14_, s_1_, s_4_, t_10_, f_3_, p_3_, j_6_, t_22_, j_4_, t_37_, t_1_, t_0_;
        p_0_ = _pp_var_A[i_];
        t_0_ = _pp_var_A[i_];
        p_1_ = _pp_var_B[i_];
        t_1_ = _pp_var_B[i_];
        p_2_ = _pp_var_C[i_];
        t_2_ = _pp_var_C[i_];
        p_3_ = _pp_var_d[i_];
        t_3_ = _pp_var_d[i_];
        p_4_ = _pp_var_e[i_];
        t_4_ = _pp_var_e[i_];
        s_0_ =  1.0/_pp_var_s0;
        s_1_ =  1.0/_pp_var_s0;
        s_2_ =  1.0/_pp_var_s0;
        s_3_ =  1.0/_pp_var_s1;
        s_4_ =  1.0/_pp_var_s1;
        f_0_ = t_0_-(p_0_+( -1.0*(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)+ -1.0*(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w))*dt*s_0_);
        f_1_ = t_1_-(p_1_+ -1.0*(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)*dt*s_1_);
        f_2_ = t_2_-(p_2_+(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)*dt*s_2_);
        f_3_ = t_3_-(p_3_+ -1.0*(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w)*dt*s_3_);
        f_4_ = t_4_-(p_4_+(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w)*dt*s_4_);
        j_0_ =  1.0-( -1.0*(t_1_*_pp_var_x)+ -1.0*(t_3_*_pp_var_z))*dt*s_0_;
        j_1_ =  -( -1.0*(t_0_*_pp_var_x)*dt*s_0_);
        j_2_ =  -( -1.0* -_pp_var_y*dt*s_0_);
        j_3_ =  -( -1.0*(t_0_*_pp_var_z)*dt*s_0_);
        j_4_ =  -( -1.0* -_pp_var_w*dt*s_0_);
        j_5_ =  -( -1.0*(t_1_*_pp_var_x)*dt*s_1_);
        j_6_ =  1.0- -1.0*(t_0_*_pp_var_x)*dt*s_1_;
        j_7_ =  -( -1.0* -_pp_var_y*dt*s_1_);
        j_8_ =  -(t_1_*_pp_var_x*dt*s_2_);
        j_9_ =  -(t_0_*_pp_var_x*dt*s_2_);
        j_10_ =  1.0- -_pp_var_y*dt*s_2_;
        j_11_ =  -( -1.0*(t_3_*_pp_var_z)*dt*s_3_);
        j_12_ =  1.0- -1.0*(t_0_*_pp_var_z)*dt*s_3_;
        j_13_ =  -( -1.0* -_pp_var_w*dt*s_3_);
        j_14_ =  -(t_3_*_pp_var_z*dt*s_4_);
        j_15_ =  -(t_0_*_pp_var_z*dt*s_4_);
        j_16_ =  1.0- -_pp_var_w*dt*s_4_;
        t_5_ = j_16_*j_0_-j_4_*j_14_;
        t_6_ = j_16_*j_1_;
        t_7_ = j_16_*j_2_;
        t_8_ = j_16_*j_3_-j_4_*j_15_;
        t_9_ = j_16_*f_0_-j_4_*f_4_;
        t_10_ = j_16_*j_11_-j_13_*j_14_;
        t_11_ = j_16_*j_12_-j_13_*j_15_;
        t_12_ = j_16_*f_3_-j_13_*f_4_;
        t_13_ = t_11_*t_5_-t_8_*t_10_;
        t_14_ = t_11_*t_6_;
        t_15_ = t_11_*t_7_;
        t_16_ = t_11_*t_9_-t_8_*t_12_;
        t_17_ = t_11_*j_14_-j_15_*t_10_;
        t_18_ = t_11_*j_16_;
        t_19_ = t_11_*f_4_-j_15_*t_12_;
        t_20_ = j_10_*t_13_-t_15_*j_8_;
        t_21_ = j_10_*t_14_-t_15_*j_9_;
        t_22_ = j_10_*t_16_-t_15_*f_2_;
        t_23_ = j_10_*j_5_-j_7_*j_8_;
        t_24_ = j_10_*j_6_-j_7_*j_9_;
        t_25_ = j_10_*f_1_-j_7_*f_2_;
        t_26_ = t_24_*t_20_-t_21_*t_23_;
        t_27_ = t_24_*t_22_-t_21_*t_25_;
        t_28_ = t_24_*j_8_-j_9_*t_23_;
        t_29_ = t_24_*j_10_;
        t_30_ = t_24_*f_2_-j_9_*t_25_;
        t_31_ = t_26_*t_24_;
        t_32_ = t_26_*t_25_-t_23_*t_27_;
        t_33_ = t_26_*t_29_;
        t_34_ = t_26_*t_30_-t_28_*t_27_;
        t_35_ = t_26_*t_11_;
        t_36_ = t_26_*t_12_-t_10_*t_27_;
        t_37_ = t_26_*t_18_;
        t_38_ = t_26_*t_19_-t_17_*t_27_;
        t_0_ = t_0_-t_27_/t_26_;
        t_1_ = t_1_-t_32_/t_31_;
        t_2_ = t_2_-t_34_/t_33_;
        t_3_ = t_3_-t_36_/t_35_;
        t_4_ = t_4_-t_38_/t_37_;
        f_0_ = t_0_-(p_0_+( -1.0*(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)+ -1.0*(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w))*dt*s_0_);
        f_1_ = t_1_-(p_1_+ -1.0*(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)*dt*s_1_);
        f_2_ = t_2_-(p_2_+(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)*dt*s_2_);
        f_3_ = t_3_-(p_3_+ -1.0*(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w)*dt*s_3_);
        f_4_ = t_4_-(p_4_+(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w)*dt*s_4_);
        j_0_ =  1.0-( -1.0*(t_1_*_pp_var_x)+ -1.0*(t_3_*_pp_var_z))*dt*s_0_;
        j_1_ =  -( -1.0*(t_0_*_pp_var_x)*dt*s_0_);
        j_2_ =  -( -1.0* -_pp_var_y*dt*s_0_);
        j_3_ =  -( -1.0*(t_0_*_pp_var_z)*dt*s_0_);
        j_4_ =  -( -1.0* -_pp_var_w*dt*s_0_);
        j_5_ =  -( -1.0*(t_1_*_pp_var_x)*dt*s_1_);
        j_6_ =  1.0- -1.0*(t_0_*_pp_var_x)*dt*s_1_;
        j_7_ =  -( -1.0* -_pp_var_y*dt*s_1_);
        j_8_ =  -(t_1_*_pp_var_x*dt*s_2_);
        j_9_ =  -(t_0_*_pp_var_x*dt*s_2_);
        j_10_ =  1.0- -_pp_var_y*dt*s_2_;
        j_11_ =  -( -1.0*(t_3_*_pp_var_z)*dt*s_3_);
        j_12_ =  1.0- -1.0*(t_0_*_pp_var_z)*dt*s_3_;
        j_13_ =  -( -1.0* -_pp_var_w*dt*s_3_);
        j_14_ =  -(t_3_*_pp_var_z*dt*s_4_);
        j_15_ =  -(t_0_*_pp_var_z*dt*s_4_);
        j_16_ =  1.0- -_pp_var_w*dt*s_4_;
        t_5_ = j_16_*j_0_-j_4_*j_14_;
        t_6_ = j_16_*j_1_;
        t_7_ = j_16_*j_2_;
        t_8_ = j_16_*j_3_-j_4_*j_15_;
        t_9_ = j_16_*f_0_-j_4_*f_4_;
        t_10_ = j_16_*j_11_-j_13_*j_14_;
        t_11_ = j_16_*j_12_-j_13_*j_15_;
        t_12_ = j_16_*f_3_-j_13_*f_4_;
        t_13_ = t_11_*t_5_-t_8_*t_10_;
        t_14_ = t_11_*t_6_;
        t_15_ = t_11_*t_7_;
        t_16_ = t_11_*t_9_-t_8_*t_12_;
        t_17_ = t_11_*j_14_-j_15_*t_10_;
        t_18_ = t_11_*j_16_;
        t_19_ = t_11_*f_4_-j_15_*t_12_;
        t_20_ = j_10_*t_13_-t_15_*j_8_;
        t_21_ = j_10_*t_14_-t_15_*j_9_;
        t_22_ = j_10_*t_16_-t_15_*f_2_;
        t_23_ = j_10_*j_5_-j_7_*j_8_;
        t_24_ = j_10_*j_6_-j_7_*j_9_;
        t_25_ = j_10_*f_1_-j_7_*f_2_;
        t_26_ = t_24_*t_20_-t_21_*t_23_;
        t_27_ = t_24_*t_22_-t_21_*t_25_;
        t_28_ = t_24_*j_8_-j_9_*t_23_;
        t_29_ = t_24_*j_10_;
        t_30_ = t_24_*f_2_-j_9_*t_25_;
        t_31_ = t_26_*t_24_;
        t_32_ = t_26_*t_25_-t_23_*t_27_;
        t_33_ = t_26_*t_29_;
        t_34_ = t_26_*t_30_-t_28_*t_27_;
        t_35_ = t_26_*t_11_;
        t_36_ = t_26_*t_12_-t_10_*t_27_;
        t_37_ = t_26_*t_18_;
        t_38_ = t_26_*t_19_-t_17_*t_27_;
        t_0_ = t_0_-t_27_/t_26_;
        t_1_ = t_1_-t_32_/t_31_;
        t_2_ = t_2_-t_34_/t_33_;
        t_3_ = t_3_-t_36_/t_35_;
        t_4_ = t_4_-t_38_/t_37_;
        f_0_ = t_0_-(p_0_+( -1.0*(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)+ -1.0*(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w))*dt*s_0_);
        f_1_ = t_1_-(p_1_+ -1.0*(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)*dt*s_1_);
        f_2_ = t_2_-(p_2_+(t_1_*(t_0_*_pp_var_x)-t_2_*_pp_var_y)*dt*s_2_);
        f_3_ = t_3_-(p_3_+ -1.0*(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w)*dt*s_3_);
        f_4_ = t_4_-(p_4_+(t_3_*(t_0_*_pp_var_z)-t_4_*_pp_var_w)*dt*s_4_);
        j_0_ =  1.0-( -1.0*(t_1_*_pp_var_x)+ -1.0*(t_3_*_pp_var_z))*dt*s_0_;
        j_1_ =  -( -1.0*(t_0_*_pp_var_x)*dt*s_0_);
        j_2_ =  -( -1.0* -_pp_var_y*dt*s_0_);
        j_3_ =  -( -1.0*(t_0_*_pp_var_z)*dt*s_0_);
        j_4_ =  -( -1.0* -_pp_var_w*dt*s_0_);
        j_5_ =  -( -1.0*(t_1_*_pp_var_x)*dt*s_1_);
        j_6_ =  1.0- -1.0*(t_0_*_pp_var_x)*dt*s_1_;
        j_7_ =  -( -1.0* -_pp_var_y*dt*s_1_);
        j_8_ =  -(t_1_*_pp_var_x*dt*s_2_);
        j_9_ =  -(t_0_*_pp_var_x*dt*s_2_);
        j_10_ =  1.0- -_pp_var_y*dt*s_2_;
        j_11_ =  -( -1.0*(t_3_*_pp_var_z)*dt*s_3_);
        j_12_ =  1.0- -1.0*(t_0_*_pp_var_z)*dt*s_3_;
        j_13_ =  -( -1.0* -_pp_var_w*dt*s_3_);
        j_14_ =  -(t_3_*_pp_var_z*dt*s_4_);
        j_15_ =  -(t_0_*_pp_var_z*dt*s_4_);
        j_16_ =  1.0- -_pp_var_w*dt*s_4_;
        t_5_ = j_16_*j_0_-j_4_*j_14_;
        t_6_ = j_16_*j_1_;
        t_7_ = j_16_*j_2_;
        t_8_ = j_16_*j_3_-j_4_*j_15_;
        t_9_ = j_16_*f_0_-j_4_*f_4_;
        t_10_ = j_16_*j_11_-j_13_*j_14_;
        t_11_ = j_16_*j_12_-j_13_*j_15_;
        t_12_ = j_16_*f_3_-j_13_*f_4_;
        t_13_ = t_11_*t_5_-t_8_*t_10_;
        t_14_ = t_11_*t_6_;
        t_15_ = t_11_*t_7_;
        t_16_ = t_11_*t_9_-t_8_*t_12_;
        t_17_ = t_11_*j_14_-j_15_*t_10_;
        t_18_ = t_11_*j_16_;
        t_19_ = t_11_*f_4_-j_15_*t_12_;
        t_20_ = j_10_*t_13_-t_15_*j_8_;
        t_21_ = j_10_*t_14_-t_15_*j_9_;
        t_22_ = j_10_*t_16_-t_15_*f_2_;
        t_23_ = j_10_*j_5_-j_7_*j_8_;
        t_24_ = j_10_*j_6_-j_7_*j_9_;
        t_25_ = j_10_*f_1_-j_7_*f_2_;
        t_26_ = t_24_*t_20_-t_21_*t_23_;
        t_27_ = t_24_*t_22_-t_21_*t_25_;
        t_28_ = t_24_*j_8_-j_9_*t_23_;
        t_29_ = t_24_*j_10_;
        t_30_ = t_24_*f_2_-j_9_*t_25_;
        t_31_ = t_26_*t_24_;
        t_32_ = t_26_*t_25_-t_23_*t_27_;
        t_33_ = t_26_*t_29_;
        t_34_ = t_26_*t_30_-t_28_*t_27_;
        t_35_ = t_26_*t_11_;
        t_36_ = t_26_*t_12_-t_10_*t_27_;
        t_37_ = t_26_*t_18_;
        t_38_ = t_26_*t_19_-t_17_*t_27_;
        t_0_ = t_0_-t_27_/t_26_;
        t_1_ = t_1_-t_32_/t_31_;
        t_2_ = t_2_-t_34_/t_33_;
        t_3_ = t_3_-t_36_/t_35_;
        t_4_ = t_4_-t_38_/t_37_;
        _pp_var_A[i_] = t_0_;
        _pp_var_B[i_] = t_1_;
        _pp_var_C[i_] = t_2_;
        _pp_var_d[i_] = t_3_;
        _pp_var_e[i_] = t_4_;
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
} // namespace kernel_test4_kin_compartment
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test4_kin_compartment_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_test4_kin_compartment::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_test4_kin_compartment::min_align_;
    result.init_mechanism = testing::kernel_test4_kin_compartment::init;
    result.compute_currents = testing::kernel_test4_kin_compartment::compute_currents;
    result.apply_events = testing::kernel_test4_kin_compartment::apply_events;
    result.advance_state = testing::kernel_test4_kin_compartment::advance_state;
    result.write_ions = testing::kernel_test4_kin_compartment::write_ions;
    result.post_event = testing::kernel_test4_kin_compartment::post_event;
    return &result;
  }}

