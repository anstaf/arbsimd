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
auto* _pp_var_a __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_b __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_c __attribute__((unused)) = params_.state_vars[2];\
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
        _pp_var_a[tid_] =  0.20000000000000001;
        _pp_var_b[tid_] =  0.29999999999999999;
        _pp_var_c[tid_] =  0.5;
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
        arb_value_type t_16_, t_15_, t_13_, t_12_, t_11_, t_9_, t_7_, t_6_, t_5_, t_4_, t_10_, p_2_, j_8_, j_4_, j_6_, f_2_, j_2_, t_3_, j_5_, t_17_, t_8_, j_1_, t_14_, j_3_, f1, f_0_, j_7_, f_1_, r0, t_2_, f0, j_0_, p_0_, t_1_, r1, p_1_, t_0_;
        p_0_ = _pp_var_a[tid_];
        t_0_ = _pp_var_a[tid_];
        p_1_ = _pp_var_b[tid_];
        t_1_ = _pp_var_b[tid_];
        p_2_ = _pp_var_c[tid_];
        t_2_ = _pp_var_c[tid_];
        f0 =  2.0;
        r0 =  1.0;
        f1 =  3.0;
        r1 =  0.;
        f_0_ = t_0_-(p_0_+ -1.0*(t_1_*(t_0_*f0)-t_2_*r0)*dt);
        f_1_ = t_1_-(p_1_+( -1.0*(t_1_*(t_0_*f0)-t_2_*r0)+(t_2_*f1-t_1_*r1))*dt);
        f_2_ = t_2_-(p_2_+(t_1_*(t_0_*f0)-t_2_*r0+ -1.0*(t_2_*f1-t_1_*r1))*dt);
        j_0_ =  1.0- -1.0*(t_1_*f0)*dt;
        j_1_ =  -( -1.0*(t_0_*f0)*dt);
        j_2_ =  -( -1.0* -r0*dt);
        j_3_ =  -( -1.0*(t_1_*f0)*dt);
        j_4_ =  1.0-( -1.0*(t_0_*f0)+ -r1)*dt;
        j_5_ =  -(( -1.0* -r0+f1)*dt);
        j_6_ =  -(t_1_*f0*dt);
        j_7_ =  -((t_0_*f0+ -1.0* -r1)*dt);
        j_8_ =  1.0-( -r0+ -1.0*f1)*dt;
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
        f_0_ = t_0_-(p_0_+ -1.0*(t_1_*(t_0_*f0)-t_2_*r0)*dt);
        f_1_ = t_1_-(p_1_+( -1.0*(t_1_*(t_0_*f0)-t_2_*r0)+(t_2_*f1-t_1_*r1))*dt);
        f_2_ = t_2_-(p_2_+(t_1_*(t_0_*f0)-t_2_*r0+ -1.0*(t_2_*f1-t_1_*r1))*dt);
        j_0_ =  1.0- -1.0*(t_1_*f0)*dt;
        j_1_ =  -( -1.0*(t_0_*f0)*dt);
        j_2_ =  -( -1.0* -r0*dt);
        j_3_ =  -( -1.0*(t_1_*f0)*dt);
        j_4_ =  1.0-( -1.0*(t_0_*f0)+ -r1)*dt;
        j_5_ =  -(( -1.0* -r0+f1)*dt);
        j_6_ =  -(t_1_*f0*dt);
        j_7_ =  -((t_0_*f0+ -1.0* -r1)*dt);
        j_8_ =  1.0-( -r0+ -1.0*f1)*dt;
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
        f_0_ = t_0_-(p_0_+ -1.0*(t_1_*(t_0_*f0)-t_2_*r0)*dt);
        f_1_ = t_1_-(p_1_+( -1.0*(t_1_*(t_0_*f0)-t_2_*r0)+(t_2_*f1-t_1_*r1))*dt);
        f_2_ = t_2_-(p_2_+(t_1_*(t_0_*f0)-t_2_*r0+ -1.0*(t_2_*f1-t_1_*r1))*dt);
        j_0_ =  1.0- -1.0*(t_1_*f0)*dt;
        j_1_ =  -( -1.0*(t_0_*f0)*dt);
        j_2_ =  -( -1.0* -r0*dt);
        j_3_ =  -( -1.0*(t_1_*f0)*dt);
        j_4_ =  1.0-( -1.0*(t_0_*f0)+ -r1)*dt;
        j_5_ =  -(( -1.0* -r0+f1)*dt);
        j_6_ =  -(t_1_*f0*dt);
        j_7_ =  -((t_0_*f0+ -1.0* -r1)*dt);
        j_8_ =  1.0-( -r0+ -1.0*f1)*dt;
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
        _pp_var_a[tid_] = t_0_;
        _pp_var_b[tid_] = t_1_;
        _pp_var_c[tid_] = t_2_;
    }
}

} // namespace

void mechanism_test3_kin_diff_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 3}, block_dim>>>(*p);
}

void mechanism_test3_kin_diff_gpu_compute_currents_(arb_mechanism_ppack* p) {}

void mechanism_test3_kin_diff_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test3_kin_diff_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_test3_kin_diff_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_test3_kin_diff_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace testing
