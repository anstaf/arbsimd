#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace allen_catalogue {
namespace kernel_Ca_HVA {

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
[[maybe_unused]] auto* _pp_var_m = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_gbar = pp->parameters[0];\
[[maybe_unused]] auto& _pp_var_ion_ca = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_ca_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type mBeta, mAlpha, hAlpha, ll0_, hBeta, ll1_;
        ll1_ =  0.;
        ll0_ =  0.;
        ll0_ =  -27.0-v;
        ll1_ =  3.7999999999999998*exprelr(ll0_* 0.26315789473684209);
        mAlpha =  0.055*ll1_;
        mBeta =  0.93999999999999995*exp(( -75.0-v)* 0.058823529411764705);
        _pp_var_m[i_] = mAlpha/(mAlpha+mBeta);
        hAlpha =  0.000457*exp(( -13.0-v)* 0.02);
        hBeta =  0.0064999999999999997/(exp(( -v- 15.0)* 0.035714285714285712)+ 1.0);
        _pp_var_h[i_] = hAlpha/(hAlpha+hBeta);
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
        arb_value_type ba_0_, a_0_, mRat, mBeta, a_1_, ll4_, mAlpha, hRat, ll2_, hBeta, hAlpha, ll5_, ll0_, ba_1_, ll1_, ll3_;
        ll5_ =  0.;
        ll4_ =  0.;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        ll0_ =  -27.0-v;
        ll1_ =  3.7999999999999998*exprelr(ll0_* 0.26315789473684209);
        mAlpha =  0.055*ll1_;
        mBeta =  0.93999999999999995*exp(( -75.0-v)* 0.058823529411764705);
        mRat = mAlpha+mBeta;
        hAlpha =  0.000457*exp(( -13.0-v)* 0.02);
        hBeta =  0.0064999999999999997/(exp(( -v- 15.0)* 0.035714285714285712)+ 1.0);
        hRat = hAlpha+hBeta;
        a_0_ =  -mRat;
        ba_0_ = mAlpha/a_0_;
        ll2_ = a_0_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_m[i_] =  -ba_0_+(_pp_var_m[i_]+ba_0_)*ll3_;
        a_1_ =  -hRat;
        ba_1_ = hAlpha/a_1_;
        ll4_ = a_1_*dt;
        ll5_ = ( 1.0+ 0.5*ll4_)/( 1.0- 0.5*ll4_);
        _pp_var_h[i_] =  -ba_1_+(_pp_var_h[i_]+ba_1_)*ll5_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type eca = _pp_var_ion_ca.reversal_potential[ion_ca_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ica = 0;
        ica = _pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_h[i_]*(v-eca);
        current_ = ica;
        conductivity_ = _pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_h[i_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_ca.current_density[ion_ca_indexi_] = fma(10.0*_pp_var_weight[i_], ica, _pp_var_ion_ca.current_density[ion_ca_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_Ca_HVA
} // namespace allen_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_allen_catalogue_Ca_HVA_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::allen_catalogue::kernel_Ca_HVA::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::allen_catalogue::kernel_Ca_HVA::min_align_;
    result.init_mechanism = arb::allen_catalogue::kernel_Ca_HVA::init;
    result.compute_currents = arb::allen_catalogue::kernel_Ca_HVA::compute_currents;
    result.apply_events = arb::allen_catalogue::kernel_Ca_HVA::apply_events;
    result.advance_state = arb::allen_catalogue::kernel_Ca_HVA::advance_state;
    result.write_ions = arb::allen_catalogue::kernel_Ca_HVA::write_ions;
    result.post_event = arb::allen_catalogue::kernel_Ca_HVA::post_event;
    return &result;
  }}

