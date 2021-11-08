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
auto _pp_var_a4 __attribute__((unused)) = params_.globals[0];\
auto* _pp_var_s __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_d __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_h __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_a0 __attribute__((unused)) = params_.state_vars[3];\
auto* _pp_var_a1 __attribute__((unused)) = params_.state_vars[4];\
auto* _pp_var_a2 __attribute__((unused)) = params_.state_vars[5];\
auto* _pp_var_a3 __attribute__((unused)) = params_.state_vars[6];\
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
        arb_value_type t_11_, t_8_, t_7_, t_6_, t_5_, t_1_, l_9_, t_4_, l_7_, l_8_, t_3_, l_6_, t_10_, l_5_, l_4_, t_9_, l_2_, l_1_, t_0_, l_3_, t_2_, l_0_;
        _pp_var_a0[tid_] =  2.5;
        _pp_var_a1[tid_] =  0.5;
        _pp_var_a2[tid_] =  3.0;
        _pp_var_a3[tid_] =  2.2999999999999998;
        l_0_ = _pp_var_a4-_pp_var_a3[tid_];
        l_1_ =  -_pp_var_a2[tid_];
        l_2_ =  0.;
        l_3_ = _pp_var_a0[tid_]- -_pp_var_a1[tid_];
        l_4_ =  -_pp_var_a0[tid_]- -_pp_var_a1[tid_];
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
        _pp_var_s[tid_] = t_4_/t_3_;
        _pp_var_d[tid_] = t_9_/t_8_;
        _pp_var_h[tid_] = t_11_/t_10_;
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
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        _pp_var_s[tid_] = _pp_var_a1[tid_];
    }
}

} // namespace

void mechanism_test_linear_init_shuffle_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 3}, block_dim>>>(*p);
}

void mechanism_test_linear_init_shuffle_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test_linear_init_shuffle_gpu_advance_state_(arb_mechanism_ppack* p) {}

void mechanism_test_linear_init_shuffle_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_test_linear_init_shuffle_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_test_linear_init_shuffle_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace testing
