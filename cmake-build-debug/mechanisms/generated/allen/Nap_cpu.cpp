#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace allen_catalogue {
namespace kernel_Nap {

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
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_v = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_g = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_celsius = pp->state_vars[3];\
[[maybe_unused]] auto* _pp_var_mInf = pp->state_vars[4];\
[[maybe_unused]] auto* _pp_var_hInf = pp->state_vars[5];\
[[maybe_unused]] auto* _pp_var_hTau = pp->state_vars[6];\
[[maybe_unused]] auto* _pp_var_hAlpha = pp->state_vars[7];\
[[maybe_unused]] auto* _pp_var_hBeta = pp->state_vars[8];\
[[maybe_unused]] auto* _pp_var_gbar = pp->parameters[0];\
[[maybe_unused]] auto& _pp_var_ion_na = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_na_index = pp->ion_states[0].index;\
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
        _pp_var_h[i_] = _pp_var_hInf[i_];
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 1; ++ix) {
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
        arb_value_type b_0_, a_0_, ll0_, ll1_;
        ll1_ =  0.;
        ll0_ =  0.;
        rates(pp, i_, v);
        a_0_ = _pp_var_hTau[i_];
        b_0_ = _pp_var_hInf[i_];
        ll0_ =  -dt/a_0_;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_h[i_] = b_0_+(_pp_var_h[i_]-b_0_)*ll1_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_na_indexi_ = _pp_var_ion_na_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type ena = _pp_var_ion_na.reversal_potential[ion_na_indexi_];
        arb_value_type ina = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        rates(pp, i_, v);
        _pp_var_g[i_] = _pp_var_gbar[i_]*_pp_var_mInf[i_]*_pp_var_h[i_];
        ina = _pp_var_g[i_]*(v-ena);
        current_ = ina;
        conductivity_ = _pp_var_g[i_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_na.current_density[ion_na_indexi_] = fma(10.0*_pp_var_weight[i_], ina, _pp_var_ion_na.current_density[ion_na_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
[[maybe_unused]] static void rates(arb_mechanism_ppack* pp, int i_, arb_value_type v) {
    PPACK_IFACE_BLOCK;
    arb_value_type qt, ll1_, ll2_, ll3_, ll0_;
    ll3_ =  0.;
    ll2_ =  0.;
    ll1_ =  0.;
    ll0_ =  0.;
    qt = pow( 2.2999999999999998, (_pp_var_celsius[i_]- 21.0)* 0.10000000000000001);
    _pp_var_mInf[i_] =  1.0/( 1.0+exp((v- -52.600000000000001)* -0.21739130434782611));
    _pp_var_hInf[i_] =  1.0/( 1.0+exp((v- -48.799999999999997)* 0.10000000000000001));
    ll0_ = v+ 17.0;
    if (abs(ll0_* 0.21598272138228941)< 9.9999999999999995e-07) {
        ll1_ =  4.6299999999999999*( 1.0-ll0_* 0.21598272138228941* 0.5);
    }
    else {
        ll1_ = ll0_/(exp(ll0_* 0.21598272138228941)- 1.0);
    }
    _pp_var_hAlpha[i_] =  2.88e-06*ll1_;
    ll2_ =  -(v+ 64.400000000000006);
    if (abs(ll2_* 0.38022813688212931)< 9.9999999999999995e-07) {
        ll3_ =  2.6299999999999999*( 1.0-ll2_* 0.38022813688212931* 0.5);
    }
    else {
        ll3_ = ll2_/(exp(ll2_* 0.38022813688212931)- 1.0);
    }
    _pp_var_hBeta[i_] =  6.9399999999999996e-06*ll3_;
    _pp_var_hTau[i_] =  1.0/(_pp_var_hAlpha[i_]+_pp_var_hBeta[i_])/qt;
}
#undef PPACK_IFACE_BLOCK
} // namespace kernel_Nap
} // namespace allen_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_allen_catalogue_Nap_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::allen_catalogue::kernel_Nap::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::allen_catalogue::kernel_Nap::min_align_;
    result.init_mechanism = arb::allen_catalogue::kernel_Nap::init;
    result.compute_currents = arb::allen_catalogue::kernel_Nap::compute_currents;
    result.apply_events = arb::allen_catalogue::kernel_Nap::apply_events;
    result.advance_state = arb::allen_catalogue::kernel_Nap::advance_state;
    result.write_ions = arb::allen_catalogue::kernel_Nap::write_ions;
    result.post_event = arb::allen_catalogue::kernel_Nap::post_event;
    return &result;
  }}

