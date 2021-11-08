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
auto _pp_var_A __attribute__((unused)) = params_.globals[0];\
auto _pp_var_B __attribute__((unused)) = params_.globals[1];\
auto* _pp_var_a __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_b __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_x __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_y __attribute__((unused)) = params_.state_vars[3];\
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
        _pp_var_b[tid_] =  1.0-_pp_var_a[tid_];
        _pp_var_x[tid_] =  0.59999999999999998;
        _pp_var_y[tid_] =  1.0-_pp_var_x[tid_];
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
        arb_value_type t_5_, t_3_, t_2_, t_1_, t_0_, a_9_, a_8_, a_7_, a_5_, a_4_, t_4_, a_3_, a_2_, a_1_, alpha1, alpha2, t_7_, beta1, t_6_, a_10_, a_6_, beta2, a_0_;
        a_0_ =  0.;
        alpha1 =  2.0;
        beta1 =  0.59999999999999998;
        alpha2 =  3.0;
        beta2 =  0.69999999999999996;
        a_1_ =  1.0/( 1.0/_pp_var_A);
        a_2_ =  1.0/( 1.0/_pp_var_A);
        a_3_ = _pp_var_A;
        a_4_ =  1.0;
        a_5_ =  1.0;
        a_6_ =  1.0;
        a_7_ = alpha1*( 1.0/_pp_var_A);
        a_8_ =  -beta1*( 1.0/_pp_var_A);
        a_9_ = alpha2;
        a_10_ =  -beta2;
        t_0_ = a_10_*a_4_-a_5_*a_9_;
        t_1_ = a_10_*a_6_-a_5_*a_0_;
        t_2_ = t_0_*a_10_;
        t_3_ = t_0_*a_0_-a_9_*t_1_;
        t_4_ = a_8_*a_1_-a_2_*a_7_;
        t_5_ = a_8_*a_3_-a_2_*a_0_;
        t_6_ = t_4_*a_8_;
        t_7_ = t_4_*a_0_-a_7_*t_5_;
        _pp_var_a[tid_] = t_5_/t_4_;
        _pp_var_b[tid_] = t_7_/t_6_;
        _pp_var_x[tid_] = t_1_/t_0_;
        _pp_var_y[tid_] = t_3_/t_2_;
    }
}

} // namespace

void mechanism_test1_kin_steadystate_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 4}, block_dim>>>(*p);
}

void mechanism_test1_kin_steadystate_gpu_compute_currents_(arb_mechanism_ppack* p) {}

void mechanism_test1_kin_steadystate_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test1_kin_steadystate_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_test1_kin_steadystate_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_test1_kin_steadystate_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace testing
