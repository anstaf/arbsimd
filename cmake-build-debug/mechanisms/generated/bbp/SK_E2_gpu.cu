#include <arbor/gpu/gpu_common.hpp>
#include <arbor/gpu/math_cu.hpp>
#include <arbor/gpu/reduce_by_key.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {
namespace bbp_catalogue {

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
auto _pp_var_zTau __attribute__((unused)) = params_.globals[0];\
auto* _pp_var_z __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_gSK_E2bar __attribute__((unused)) = params_.parameters[0];\
auto& _pp_var_ion_k __attribute__((unused)) = params_.ion_states[0];\
auto* _pp_var_ion_k_index __attribute__((unused)) = params_.ion_states[0].index;\
auto& _pp_var_ion_ca __attribute__((unused)) = params_.ion_states[1];\
auto* _pp_var_ion_ca_index __attribute__((unused)) = params_.ion_states[1].index;\
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
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[tid_];
        arb_value_type cai = _pp_var_ion_ca.internal_concentration[ion_ca_indexi_];
        if (cai< 9.9999999999999995e-08) {
            _pp_var_z[tid_] =  0.;
        }
        else {
            _pp_var_z[tid_] =  1.0/( 1.0+pow( 0.00042999999999999999/cai,  4.7999999999999998));
        }
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
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[tid_];
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type cai = _pp_var_ion_ca.internal_concentration[ion_ca_indexi_];
        arb_value_type a_0_, ll0_, ll2_, ll1_, b_0_;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        if (cai< 9.9999999999999995e-08) {
            ll0_ =  0.;
        }
        else {
            ll0_ =  1.0/( 1.0+pow( 0.00042999999999999999/cai,  4.7999999999999998));
        }
        a_0_ = _pp_var_zTau;
        b_0_ = ll0_;
        ll1_ =  -dt/a_0_;
        ll2_ = ( 1.0+ 0.5*ll1_)/( 1.0- 0.5*ll1_);
        _pp_var_z[tid_] = b_0_+(_pp_var_z[tid_]-b_0_)*ll2_;
    }
}

__global__
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto ion_k_indexi_ = _pp_var_ion_k_index[tid_];
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type ek = _pp_var_ion_k.reversal_potential[ion_k_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ik = 0;
        ik = _pp_var_gSK_E2bar[tid_]*_pp_var_z[tid_]*(v-ek);
        current_ = ik;
        conductivity_ = _pp_var_gSK_E2bar[tid_]*_pp_var_z[tid_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[tid_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[tid_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_k.current_density[ion_k_indexi_] = fma(10.0*_pp_var_weight[tid_], ik, _pp_var_ion_k.current_density[ion_k_indexi_]);
    }
}

} // namespace

void mechanism_SK_E2_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 1}, block_dim>>>(*p);
}

void mechanism_SK_E2_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_SK_E2_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_SK_E2_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_SK_E2_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_SK_E2_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace bbp_catalogue
} // namespace arb
