#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace default_catalogue {
namespace kernel_nax {

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
[[maybe_unused]] auto _pp_var_tha = pp->globals[0];\
[[maybe_unused]] auto _pp_var_qa = pp->globals[1];\
[[maybe_unused]] auto _pp_var_Ra = pp->globals[2];\
[[maybe_unused]] auto _pp_var_Rb = pp->globals[3];\
[[maybe_unused]] auto _pp_var_thi1 = pp->globals[4];\
[[maybe_unused]] auto _pp_var_thi2 = pp->globals[5];\
[[maybe_unused]] auto _pp_var_qd = pp->globals[6];\
[[maybe_unused]] auto _pp_var_qg = pp->globals[7];\
[[maybe_unused]] auto _pp_var_mmin = pp->globals[8];\
[[maybe_unused]] auto _pp_var_hmin = pp->globals[9];\
[[maybe_unused]] auto _pp_var_q10 = pp->globals[10];\
[[maybe_unused]] auto _pp_var_Rg = pp->globals[11];\
[[maybe_unused]] auto _pp_var_Rd = pp->globals[12];\
[[maybe_unused]] auto _pp_var_thinf = pp->globals[13];\
[[maybe_unused]] auto _pp_var_qinf = pp->globals[14];\
[[maybe_unused]] auto* _pp_var_m = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_v = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_thegna = pp->state_vars[3];\
[[maybe_unused]] auto* _pp_var_minf = pp->state_vars[4];\
[[maybe_unused]] auto* _pp_var_hinf = pp->state_vars[5];\
[[maybe_unused]] auto* _pp_var_mtau = pp->state_vars[6];\
[[maybe_unused]] auto* _pp_var_htau = pp->state_vars[7];\
[[maybe_unused]] auto* _pp_var_sh = pp->parameters[0];\
[[maybe_unused]] auto* _pp_var_gbar = pp->parameters[1];\
[[maybe_unused]] auto& _pp_var_ion_na = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_na_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes
[[maybe_unused]] static void trates(arb_mechanism_ppack* pp, int i_, arb_value_type vm, arb_value_type sh2, arb_value_type celsius);

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        trates(pp, i_, v, _pp_var_sh[i_], celsius);
        _pp_var_m[i_] = _pp_var_minf[i_];
        _pp_var_h[i_] = _pp_var_hinf[i_];
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
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type a_1_, a_0_, b_0_, ll0_, ll1_, ll2_, b_1_, ll3_;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        trates(pp, i_, v, _pp_var_sh[i_], celsius);
        a_0_ = _pp_var_mtau[i_];
        b_0_ = _pp_var_minf[i_];
        ll0_ =  -dt/a_0_;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[i_] = b_0_+(_pp_var_m[i_]-b_0_)*ll1_;
        a_1_ = _pp_var_htau[i_];
        b_1_ = _pp_var_hinf[i_];
        ll2_ =  -dt/a_1_;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_h[i_] = b_1_+(_pp_var_h[i_]-b_1_)*ll3_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_na_indexi_ = _pp_var_ion_na_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ena = _pp_var_ion_na.reversal_potential[ion_na_indexi_];
        arb_value_type ina = 0;
        _pp_var_thegna[i_] = _pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_h[i_];
        ina = _pp_var_thegna[i_]*(v-ena);
        current_ = ina;
        conductivity_ = _pp_var_thegna[i_];
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
[[maybe_unused]] static void trates(arb_mechanism_ppack* pp, int i_, arb_value_type vm, arb_value_type sh2, arb_value_type celsius) {
    PPACK_IFACE_BLOCK;
    arb_value_type b, a, ll5_, ll0_, ll1_, ll3_, ll4_, ll2_, qt;
    ll5_ =  0.;
    ll4_ =  0.;
    ll3_ =  0.;
    ll2_ =  0.;
    ll1_ =  0.;
    ll0_ =  0.;
    qt = pow(_pp_var_q10, (celsius- 24.0)* 0.10000000000000001);
    ll0_ = _pp_var_tha+sh2;
    a = _pp_var_Ra*_pp_var_qa*exprelr( -(vm-ll0_)/_pp_var_qa);
    ll1_ =  -vm;
    ll2_ =  -_pp_var_tha-sh2;
    b = _pp_var_Rb*_pp_var_qa*exprelr( -(ll1_-ll2_)/_pp_var_qa);
    _pp_var_mtau[i_] = max( 1.0/(a+b)/qt, _pp_var_mmin);
    _pp_var_minf[i_] = a/(a+b);
    ll3_ = _pp_var_thi1+sh2;
    a = _pp_var_Rd*_pp_var_qd*exprelr( -(vm-ll3_)/_pp_var_qd);
    ll4_ =  -vm;
    ll5_ =  -_pp_var_thi2-sh2;
    b = _pp_var_Rg*_pp_var_qg*exprelr( -(ll4_-ll5_)/_pp_var_qg);
    _pp_var_htau[i_] = max( 1.0/(a+b)/qt, _pp_var_hmin);
    _pp_var_hinf[i_] =  1.0/( 1.0+exp((vm-_pp_var_thinf-sh2)/_pp_var_qinf));
}
#undef PPACK_IFACE_BLOCK
} // namespace kernel_nax
} // namespace default_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_default_catalogue_nax_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::default_catalogue::kernel_nax::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::default_catalogue::kernel_nax::min_align_;
    result.init_mechanism = arb::default_catalogue::kernel_nax::init;
    result.compute_currents = arb::default_catalogue::kernel_nax::compute_currents;
    result.apply_events = arb::default_catalogue::kernel_nax::apply_events;
    result.advance_state = arb::default_catalogue::kernel_nax::advance_state;
    result.write_ions = arb::default_catalogue::kernel_nax::write_ions;
    result.post_event = arb::default_catalogue::kernel_nax::post_event;
    return &result;
  }}

