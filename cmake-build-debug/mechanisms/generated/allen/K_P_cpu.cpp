#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace allen_catalogue {
namespace kernel_K_P {

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
[[maybe_unused]] auto _pp_var_vshift = pp->globals[0];\
[[maybe_unused]] auto _pp_var_tauF = pp->globals[1];\
[[maybe_unused]] auto* _pp_var_m = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_v = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_g = pp->state_vars[3];\
[[maybe_unused]] auto* _pp_var_celsius = pp->state_vars[4];\
[[maybe_unused]] auto* _pp_var_mInf = pp->state_vars[5];\
[[maybe_unused]] auto* _pp_var_mTau = pp->state_vars[6];\
[[maybe_unused]] auto* _pp_var_hInf = pp->state_vars[7];\
[[maybe_unused]] auto* _pp_var_hTau = pp->state_vars[8];\
[[maybe_unused]] auto* _pp_var_gbar = pp->parameters[0];\
[[maybe_unused]] auto& _pp_var_ion_k = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_k_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes
[[maybe_unused]] static void rates(arb_mechanism_ppack* pp, int i_, arb_value_type v);

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        rates(pp, i_, v);
        _pp_var_m[i_] = _pp_var_mInf[i_];
        _pp_var_h[i_] = _pp_var_hInf[i_];
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 2; ++ix) {
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
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type a_1_, a_0_, b_0_, ll0_, ll1_, ll2_, b_1_, ll3_;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        rates(pp, i_, v);
        a_0_ = _pp_var_mTau[i_];
        b_0_ = _pp_var_mInf[i_];
        ll0_ =  -dt/a_0_;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[i_] = b_0_+(_pp_var_m[i_]-b_0_)*ll1_;
        a_1_ = _pp_var_hTau[i_];
        b_1_ = _pp_var_hInf[i_];
        ll2_ =  -dt/a_1_;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_h[i_] = b_1_+(_pp_var_h[i_]-b_1_)*ll3_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_k_indexi_ = _pp_var_ion_k_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type ek = _pp_var_ion_k.reversal_potential[ion_k_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ik = 0;
        _pp_var_g[i_] = _pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_h[i_];
        ik = _pp_var_g[i_]*(v-ek);
        current_ = ik;
        conductivity_ = _pp_var_g[i_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_k.current_density[ion_k_indexi_] = fma(10.0*_pp_var_weight[i_], ik, _pp_var_ion_k.current_density[ion_k_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
[[maybe_unused]] static void rates(arb_mechanism_ppack* pp, int i_, arb_value_type v) {
    PPACK_IFACE_BLOCK;
    arb_value_type qt;
    qt = pow( 2.2999999999999998, (_pp_var_celsius[i_]- 21.0)* 0.10000000000000001);
    _pp_var_mInf[i_] =  1.0/( 1.0+exp( -(v-( -14.300000000000001+_pp_var_vshift))* 0.068493150684931503));
    if (v< -50.0+_pp_var_vshift) {
        _pp_var_mTau[i_] = _pp_var_tauF*( 1.25+ 175.03*exp( -(v-_pp_var_vshift)* -0.025999999999999999))/qt;
    }
    else {
        _pp_var_mTau[i_] = _pp_var_tauF*( 1.25+ 13.0*exp( -(v-_pp_var_vshift)* 0.025999999999999999))/qt;
    }
    _pp_var_hInf[i_] =  1.0/( 1.0+exp( -(v-( -54.0+_pp_var_vshift))* -0.090909090909090912));
    _pp_var_hTau[i_] = ( 360.0+( 1010.0+ 24.0*(v-( -55.0+_pp_var_vshift)))*exp(pow( -((v-( -75.0+_pp_var_vshift))* 0.020833333333333332),  2.0)))/qt;
}
#undef PPACK_IFACE_BLOCK
} // namespace kernel_K_P
} // namespace allen_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_allen_catalogue_K_P_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::allen_catalogue::kernel_K_P::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::allen_catalogue::kernel_K_P::min_align_;
    result.init_mechanism = arb::allen_catalogue::kernel_K_P::init;
    result.compute_currents = arb::allen_catalogue::kernel_K_P::compute_currents;
    result.apply_events = arb::allen_catalogue::kernel_K_P::apply_events;
    result.advance_state = arb::allen_catalogue::kernel_K_P::advance_state;
    result.write_ions = arb::allen_catalogue::kernel_K_P::write_ions;
    result.post_event = arb::allen_catalogue::kernel_K_P::post_event;
    return &result;
  }}

