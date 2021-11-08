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
auto* _pp_var_m __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_h __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_gCa_LVAstbar __attribute__((unused)) = params_.parameters[0];\
auto& _pp_var_ion_ca __attribute__((unused)) = params_.ion_states[0];\
auto* _pp_var_ion_ca_index __attribute__((unused)) = params_.ion_states[0].index;\
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
        _pp_var_m[tid_] =  1.0/( 1.0+exp( -(v+ 40.0)* 0.16666666666666666));
        _pp_var_h[tid_] =  1.0/( 1.0+exp((v+ 90.0)* 0.15625));
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
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type a_1_, ba_0_, a_0_, qt, ba_1_, ll3_, ll1_, mTau, mInf, hTau, hInf, ll0_, ll2_;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        qt =  2.952882641412121;
        mInf =  1.0/( 1.0+exp( -(v+ 40.0)* 0.16666666666666666));
        hInf =  1.0/( 1.0+exp((v+ 90.0)* 0.15625));
        mTau =  5.0+ 20.0/( 1.0+exp((v+ 35.0)* 0.20000000000000001));
        hTau =  20.0+ 50.0/( 1.0+exp((v+ 50.0)* 0.14285714285714285));
        a_0_ =  -1.0*qt/mTau;
        ba_0_ = mInf*qt/mTau/a_0_;
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[tid_] =  -ba_0_+(_pp_var_m[tid_]+ba_0_)*ll1_;
        a_1_ =  -1.0*qt/hTau;
        ba_1_ = hInf*qt/hTau/a_1_;
        ll2_ = a_1_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_h[tid_] =  -ba_1_+(_pp_var_h[tid_]+ba_1_)*ll3_;
    }
}

__global__
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[tid_];
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type eca = _pp_var_ion_ca.reversal_potential[ion_ca_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ica = 0;
        ica = _pp_var_gCa_LVAstbar[tid_]*_pp_var_m[tid_]*_pp_var_m[tid_]*_pp_var_h[tid_]*(v-eca);
        current_ = ica;
        conductivity_ = _pp_var_gCa_LVAstbar[tid_]*_pp_var_m[tid_]*_pp_var_m[tid_]*_pp_var_h[tid_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[tid_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[tid_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_ca.current_density[ion_ca_indexi_] = fma(10.0*_pp_var_weight[tid_], ica, _pp_var_ion_ca.current_density[ion_ca_indexi_]);
    }
}

} // namespace

void mechanism_Ca_LVAst_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 2}, block_dim>>>(*p);
}

void mechanism_Ca_LVAst_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_Ca_LVAst_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_Ca_LVAst_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_Ca_LVAst_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_Ca_LVAst_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace bbp_catalogue
} // namespace arb
