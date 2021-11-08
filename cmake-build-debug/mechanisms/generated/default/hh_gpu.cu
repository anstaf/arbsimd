#include <arbor/gpu/gpu_common.hpp>
#include <arbor/gpu/math_cu.hpp>
#include <arbor/gpu/reduce_by_key.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {
namespace default_catalogue {

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
auto* _pp_var_m __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_h __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_n __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_q10 __attribute__((unused)) = params_.state_vars[3];\
auto* _pp_var_gnabar __attribute__((unused)) = params_.parameters[0];\
auto* _pp_var_gkbar __attribute__((unused)) = params_.parameters[1];\
auto* _pp_var_gl __attribute__((unused)) = params_.parameters[2];\
auto* _pp_var_el __attribute__((unused)) = params_.parameters[3];\
auto& _pp_var_ion_na __attribute__((unused)) = params_.ion_states[0];\
auto* _pp_var_ion_na_index __attribute__((unused)) = params_.ion_states[0].index;\
auto& _pp_var_ion_k __attribute__((unused)) = params_.ion_states[1];\
auto* _pp_var_ion_k_index __attribute__((unused)) = params_.ion_states[1].index;\
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
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        arb_value_type r_3_, r_2_, r_1_, r_0_, beta, alpha;
        _pp_var_q10[tid_] = pow( 3.0, (celsius- 6.2999999999999998)* 0.10000000000000001);
        r_0_ =  0.;
        r_1_ =  0.;
        r_1_ =  -(v+ 40.0);
        r_0_ =  10.0*exprelr(r_1_* 0.10000000000000001);
        alpha =  0.10000000000000001*r_0_;
        beta =  4.0*exp( -(v+ 65.0)* 0.055555555555555552);
        _pp_var_m[tid_] = alpha/(alpha+beta);
        alpha =  0.070000000000000007*exp( -(v+ 65.0)* 0.050000000000000003);
        beta =  1.0/(exp( -(v+ 35.0)* 0.10000000000000001)+ 1.0);
        _pp_var_h[tid_] = alpha/(alpha+beta);
        r_2_ =  0.;
        r_3_ =  0.;
        r_3_ =  -(v+ 55.0);
        r_2_ =  10.0*exprelr(r_3_* 0.10000000000000001);
        alpha =  0.01*r_2_;
        beta =  0.125*exp( -(v+ 65.0)* 0.012500000000000001);
        _pp_var_n[tid_] = alpha/(alpha+beta);
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
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type a_2_, r_2_, ba_0_, a_0_, alpha, ll5_, ll0_, beta, ba_1_, ll3_, ll1_, ba_2_, r_3_, ll2_, r_0_, r_1_, sum, a_1_, ll4_;
        ll5_ =  0.;
        ll4_ =  0.;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        r_0_ =  0.;
        r_1_ =  0.;
        r_1_ =  -(v+ 40.0);
        r_0_ =  10.0*exprelr(r_1_* 0.10000000000000001);
        alpha =  0.10000000000000001*r_0_;
        beta =  4.0*exp( -(v+ 65.0)* 0.055555555555555552);
        sum = alpha+beta;
        a_0_ =  -sum*_pp_var_q10[tid_];
        ba_0_ = alpha*_pp_var_q10[tid_]/a_0_;
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[tid_] =  -ba_0_+(_pp_var_m[tid_]+ba_0_)*ll1_;
        alpha =  0.070000000000000007*exp( -(v+ 65.0)* 0.050000000000000003);
        beta =  1.0/(exp( -(v+ 35.0)* 0.10000000000000001)+ 1.0);
        sum = alpha+beta;
        a_1_ =  -sum*_pp_var_q10[tid_];
        ba_1_ = alpha*_pp_var_q10[tid_]/a_1_;
        ll2_ = a_1_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_h[tid_] =  -ba_1_+(_pp_var_h[tid_]+ba_1_)*ll3_;
        r_2_ =  0.;
        r_3_ =  0.;
        r_3_ =  -(v+ 55.0);
        r_2_ =  10.0*exprelr(r_3_* 0.10000000000000001);
        alpha =  0.01*r_2_;
        beta =  0.125*exp( -(v+ 65.0)* 0.012500000000000001);
        sum = alpha+beta;
        a_2_ =  -sum*_pp_var_q10[tid_];
        ba_2_ = alpha*_pp_var_q10[tid_]/a_2_;
        ll4_ = a_2_*dt;
        ll5_ = ( 1.0+ 0.5*ll4_)/( 1.0- 0.5*ll4_);
        _pp_var_n[tid_] =  -ba_2_+(_pp_var_n[tid_]+ba_2_)*ll5_;
    }
}

__global__
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto ion_k_indexi_ = _pp_var_ion_k_index[tid_];
        auto ion_na_indexi_ = _pp_var_ion_na_index[tid_];
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type conductivity_ = 0;
        arb_value_type ina = 0;
        arb_value_type current_ = 0;
        arb_value_type ek = _pp_var_ion_k.reversal_potential[ion_k_indexi_];
        arb_value_type ik = 0;
        arb_value_type ena = _pp_var_ion_na.reversal_potential[ion_na_indexi_];
        arb_value_type il = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type gk, n2, m_, n_;
        n_ = _pp_var_n[tid_];
        m_ = _pp_var_m[tid_];
        n2 = n_*n_;
        gk = _pp_var_gkbar[tid_]*n2*n2;
        ina = _pp_var_gnabar[tid_]*m_*m_*m_*_pp_var_h[tid_]*(v-ena);
        ik = gk*(v-ek);
        il = _pp_var_gl[tid_]*(v-_pp_var_el[tid_]);
        current_ = ik+il+ina;
        conductivity_ = _pp_var_gnabar[tid_]*m_*m_*m_*_pp_var_h[tid_]+gk+_pp_var_gl[tid_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[tid_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_ion_na.current_density[ion_na_indexi_] = fma(10.0*_pp_var_weight[tid_], ina, _pp_var_ion_na.current_density[ion_na_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[tid_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_k.current_density[ion_k_indexi_] = fma(10.0*_pp_var_weight[tid_], ik, _pp_var_ion_k.current_density[ion_k_indexi_]);
    }
}

} // namespace

void mechanism_hh_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 3}, block_dim>>>(*p);
}

void mechanism_hh_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_hh_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_hh_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_hh_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_hh_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace default_catalogue
} // namespace arb
