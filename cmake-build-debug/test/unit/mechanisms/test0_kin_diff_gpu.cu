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
auto* _pp_var_s __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_d __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_h __attribute__((unused)) = params_.state_vars[2];\
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
        _pp_var_h[tid_] =  0.20000000000000001;
        _pp_var_d[tid_] =  0.29999999999999999;
        _pp_var_s[tid_] =  1.0-_pp_var_d[tid_]-_pp_var_h[tid_];
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
        arb_value_type t_8_, t_5_, t_3_, t_2_, t_1_, t_0_, a_5_, a_4_, t_4_, a_3_, a_2_, a_1_, a_0_, alpha1, alpha2, t_7_, beta1, t_6_, a_6_, beta2;
        alpha1 =  2.0;
        beta1 =  0.59999999999999998;
        alpha2 =  3.0;
        beta2 =  0.69999999999999996;
        a_0_ =  1.0- -1.0*alpha2*dt;
        a_1_ =  -( -1.0* -beta2*dt);
        a_2_ =  1.0- -beta1*dt;
        a_3_ =  -(alpha1*dt);
        a_4_ =  -(alpha2*dt);
        a_5_ =  -( -1.0* -beta1*dt);
        a_6_ =  1.0-( -1.0*alpha1+ -beta2)*dt;
        t_0_ = a_2_*a_4_;
        t_1_ = a_2_*a_6_-a_5_*a_3_;
        t_2_ = a_2_*_pp_var_s[tid_]-a_5_*_pp_var_h[tid_];
        t_3_ = a_0_*t_1_-t_0_*a_1_;
        t_4_ = a_0_*t_2_-t_0_*_pp_var_d[tid_];
        t_5_ = t_3_*a_0_;
        t_6_ = t_3_*_pp_var_d[tid_]-a_1_*t_4_;
        t_7_ = t_3_*a_2_;
        t_8_ = t_3_*_pp_var_h[tid_]-a_3_*t_4_;
        _pp_var_d[tid_] = t_6_/t_5_;
        _pp_var_h[tid_] = t_8_/t_7_;
        _pp_var_s[tid_] = t_4_/t_3_;
    }
}

} // namespace

void mechanism_test0_kin_diff_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 3}, block_dim>>>(*p);
}

void mechanism_test0_kin_diff_gpu_compute_currents_(arb_mechanism_ppack* p) {}

void mechanism_test0_kin_diff_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test0_kin_diff_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_test0_kin_diff_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_test0_kin_diff_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace testing
