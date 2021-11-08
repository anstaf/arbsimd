#include <arbor/gpu/gpu_common.hpp>
#include <arbor/gpu/math_cu.hpp>
#include <arbor/gpu/reduce_by_key.hpp>
#include <arbor/mechanism_abi.h>

namespace testing {

#define PPACK_IFACE_BLOCK \
auto  _pp_var_width             __attribute__((unused)) = params_.width;\
auto  _pp_var_n_detectors       __attribute__((unused)) = params_.n_detectors;\
auto* _pp_var_vec_ci            __attribute__((unused)) = params_.vec_ci;\
auto* _pp_var_vec_di            __attribute__((unused)) = params_.vec_di;\
auto* _pp_var_vec_t             __attribute__((unused)) = params_.vec_t;\
auto* _pp_var_vec_dt            __attribute__((unused)) = params_.vec_dt;\
auto* _pp_var_vec_v             __attribute__((unused)) = params_.vec_v;\
auto* _pp_var_vec_i             __attribute__((unused)) = params_.vec_i;\
auto* _pp_var_vec_g             __attribute__((unused)) = params_.vec_g;\
auto* _pp_var_temperature_degC  __attribute__((unused)) = params_.temperature_degC;\
auto* _pp_var_diam_um           __attribute__((unused)) = params_.diam_um;\
auto* _pp_var_time_since_spike  __attribute__((unused)) = params_.time_since_spike;\
auto* _pp_var_node_index        __attribute__((unused)) = params_.node_index;\
auto* _pp_var_peer_index        __attribute__((unused)) = params_.peer_index;\
auto* _pp_var_multiplicity      __attribute__((unused)) = params_.multiplicity;\
auto* _pp_var_state_vars        __attribute__((unused)) = params_.state_vars;\
auto* _pp_var_weight            __attribute__((unused)) = params_.weight;\
auto& _pp_var_events            __attribute__((unused)) = params_.events;\
auto& _pp_var_mechanism_id      __attribute__((unused)) = params_.mechanism_id;\
auto& _pp_var_index_constraints __attribute__((unused)) = params_.index_constraints;\
auto _pp_var_x __attribute__((unused)) = params_.globals[0];\
auto _pp_var_y __attribute__((unused)) = params_.globals[1];\
auto _pp_var_z __attribute__((unused)) = params_.globals[2];\
auto _pp_var_w __attribute__((unused)) = params_.globals[3];\
auto _pp_var_s0 __attribute__((unused)) = params_.globals[4];\
auto _pp_var_s1 __attribute__((unused)) = params_.globals[5];\
auto* _pp_var_A __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_B __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_C __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_d __attribute__((unused)) = params_.state_vars[3];\
auto* _pp_var_e __attribute__((unused)) = params_.state_vars[4];\
//End of IFACEBLOCK

namespace {

using ::arb::gpu::exprelr;
using ::arb::gpu::safeinv;
using ::arb::gpu::min;
using ::arb::gpu::max;

__global__
void init(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        _pp_var_A[tid_] =  4.5;
        _pp_var_B[tid_] =  6.5999999999999996;
        _pp_var_C[tid_] =  0.28000000000000003;
        _pp_var_d[tid_] =  2.0;
        _pp_var_e[tid_] =  0.;
    }
}

__global__
void multiply(arb_mechanism_ppack params_) {
    PPACK_IFACE_BLOCK;
    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    auto idx_ = blockIdx.y;    if(tid_<_pp_var_width) {
        _pp_var_state_vars[idx_][tid_] *= _pp_var_multiplicity[tid_];
    }
}

__global__
void advance_state(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type t_38_, t_36_, t_35_, t_34_, t_33_, t_31_, t_29_, t_26_, t_25_, t_24_, t_23_, t_21_, t_19_, t_16_, t_15_, t_13_, t_28_, t_11_, j_10_, t_12_, t_9_, t_18_, t_7_, t_6_, t_5_, j_16_, t_3_, j_15_, f_0_, f_4_, p_2_, j_9_, t_2_, f_2_, t_17_, j_7_, j_8_, j_12_, f_1_, t_30_, j_2_, p_1_, t_27_, s_3_, s_2_, s_0_, t_32_, t_14_, j_11_, j_3_, j_13_, t_20_, j_0_, p_0_, p_4_, j_5_, t_8_, j_1_, t_4_, j_14_, s_1_, s_4_, t_10_, f_3_, p_3_, j_6_, t_22_, j_4_, t_37_, t_1_, t_0_;
        p_0_ = _pp_var_A[tid_];
        t_0_ = _pp_var_A[tid_];
        p_1_ = _pp_var_B[tid_];
        t_1_ = _pp_var_B[tid_];
        p_2_ = _pp_var_C[tid_];
        t_2_ = _pp_var_C[tid_];
        p_3_ = _pp_var_d[tid_];
        t_3_ = _pp_var_d[tid_];
        p_4_ = _pp_var_e[tid_];
        t_4_ = _pp_var_e[tid_];
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
        _pp_var_A[tid_] = t_0_;
        _pp_var_B[tid_] = t_1_;
        _pp_var_C[tid_] = t_2_;
        _pp_var_d[tid_] = t_3_;
        _pp_var_e[tid_] = t_4_;
    }
}

} // namespace

void mechanism_test4_kin_compartment_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 5}, block_dim>>>(*p);
}

void mechanism_test4_kin_compartment_gpu_compute_currents_(arb_mechanism_ppack* p) {}

void mechanism_test4_kin_compartment_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test4_kin_compartment_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_test4_kin_compartment_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_test4_kin_compartment_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace testing
