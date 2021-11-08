#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace default_catalogue {
namespace kernel_exp2syn {

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
[[maybe_unused]] auto* _pp_var_A = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_B = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_factor = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_tau1 = pp->parameters[0];\
[[maybe_unused]] auto* _pp_var_tau2 = pp->parameters[1];\
[[maybe_unused]] auto* _pp_var_e = pp->parameters[2];\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        arb_value_type tp;
        _pp_var_A[i_] =  0.;
        _pp_var_B[i_] =  0.;
        tp = _pp_var_tau1[i_]*_pp_var_tau2[i_]/(_pp_var_tau2[i_]-_pp_var_tau1[i_])*log(_pp_var_tau2[i_]/_pp_var_tau1[i_]);
        _pp_var_factor[i_] =  1.0/( -exp( -tp/_pp_var_tau1[i_])+exp( -tp/_pp_var_tau2[i_]));
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
        arb_value_type a_1_, a_0_, ll0_, ll1_, ll2_, ll3_;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        a_0_ =  -1.0/_pp_var_tau1[i_];
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_A[i_] = _pp_var_A[i_]*ll1_;
        a_1_ =  -1.0/_pp_var_tau2[i_];
        ll2_ = a_1_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_B[i_] = _pp_var_B[i_]*ll3_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type current_ = 0;
        arb_value_type i = 0;
        i = (_pp_var_B[i_]-_pp_var_A[i_])*(v-_pp_var_e[i_]);
        current_ = i;
        conductivity_ = _pp_var_B[i_]-_pp_var_A[i_];
        _pp_var_vec_g[node_indexi_] = fma(_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack* pp, arb_deliverable_event_stream* stream_ptr) {
    PPACK_IFACE_BLOCK;
    auto ncell = stream_ptr->n_streams;
    for (arb_size_type c = 0; c<ncell; ++c) {
        auto begin  = stream_ptr->events + stream_ptr->begin[c];
        auto end    = stream_ptr->events + stream_ptr->end[c];
        for (auto p = begin; p<end; ++p) {
            auto i_     = p->mech_index;
            auto weight = p->weight;
            if (p->mech_id==_pp_var_mechanism_id) {
                _pp_var_A[i_] = _pp_var_A[i_]+weight*_pp_var_factor[i_];
                _pp_var_B[i_] = _pp_var_B[i_]+weight*_pp_var_factor[i_];
            }
        }
    }
}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_exp2syn
} // namespace default_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_default_catalogue_exp2syn_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::default_catalogue::kernel_exp2syn::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::default_catalogue::kernel_exp2syn::min_align_;
    result.init_mechanism = arb::default_catalogue::kernel_exp2syn::init;
    result.compute_currents = arb::default_catalogue::kernel_exp2syn::compute_currents;
    result.apply_events = arb::default_catalogue::kernel_exp2syn::apply_events;
    result.advance_state = arb::default_catalogue::kernel_exp2syn::advance_state;
    result.write_ions = arb::default_catalogue::kernel_exp2syn::write_ions;
    result.post_event = arb::default_catalogue::kernel_exp2syn::post_event;
    return &result;
  }}

