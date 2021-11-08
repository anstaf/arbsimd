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
auto _pp_var_F __attribute__((unused)) = params_.globals[0];\
auto* _pp_var_cai __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_gamma __attribute__((unused)) = params_.parameters[0];\
auto* _pp_var_decay __attribute__((unused)) = params_.parameters[1];\
auto* _pp_var_depth __attribute__((unused)) = params_.parameters[2];\
auto* _pp_var_minCai __attribute__((unused)) = params_.parameters[3];\
auto* _pp_var_initCai __attribute__((unused)) = params_.parameters[4];\
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
        _pp_var_cai[tid_] = _pp_var_initCai[tid_];
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
        arb_value_type ica = 0.10000000000000001*_pp_var_ion_ca.current_density[ion_ca_indexi_];
        arb_value_type ll0_, ba_0_, a_0_, ll1_;
        ll1_ =  0.;
        ll0_ =  0.;
        a_0_ =  -( 1.0/_pp_var_decay[tid_]);
        ba_0_ = ( -5000.0*ica*_pp_var_gamma[tid_]/(_pp_var_F*_pp_var_depth[tid_])- -_pp_var_minCai[tid_]/_pp_var_decay[tid_])/a_0_;
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_cai[tid_] =  -ba_0_+(_pp_var_cai[tid_]+ba_0_)*ll1_;
    }
}

__global__
void write_ions(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[tid_];
        arb_value_type cai_shadowed_ = 0;
        cai_shadowed_ = _pp_var_cai[tid_];
        _pp_var_ion_ca.internal_concentration[ion_ca_indexi_] = fma(_pp_var_weight[tid_], cai_shadowed_, _pp_var_ion_ca.internal_concentration[ion_ca_indexi_]);
    }
}

} // namespace

void mechanism_CaDynamics_E2_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 1}, block_dim>>>(*p);
}

void mechanism_CaDynamics_E2_gpu_compute_currents_(arb_mechanism_ppack* p) {}

void mechanism_CaDynamics_E2_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_CaDynamics_E2_gpu_write_ions_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    write_ions<<<grid_dim, block_dim>>>(*p);
}

void mechanism_CaDynamics_E2_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_CaDynamics_E2_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace bbp_catalogue
} // namespace arb
