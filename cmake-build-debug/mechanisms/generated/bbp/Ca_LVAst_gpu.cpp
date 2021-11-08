#include <arbor/mechanism_abi.h>
#include <cmath>

namespace arb {
namespace bbp_catalogue {
void mechanism_Ca_LVAst_gpu_init_(arb_mechanism_ppack*);
void mechanism_Ca_LVAst_gpu_advance_state_(arb_mechanism_ppack*);
void mechanism_Ca_LVAst_gpu_compute_currents_(arb_mechanism_ppack*);
void mechanism_Ca_LVAst_gpu_write_ions_(arb_mechanism_ppack*);
void mechanism_Ca_LVAst_gpu_apply_events_(arb_mechanism_ppack*, arb_deliverable_event_stream*);
void mechanism_Ca_LVAst_gpu_post_event_(arb_mechanism_ppack*);

} // namespace bbp_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_bbp_catalogue_Ca_LVAst_interface_gpu() {
    static arb_mechanism_interface result;
    result.backend=arb_backend_kind_gpu;
    result.partition_width=1;
    result.alignment=1;
    result.init_mechanism=arb::bbp_catalogue::mechanism_Ca_LVAst_gpu_init_;
    result.compute_currents=arb::bbp_catalogue::mechanism_Ca_LVAst_gpu_compute_currents_;
    result.apply_events=arb::bbp_catalogue::mechanism_Ca_LVAst_gpu_apply_events_;
    result.advance_state=arb::bbp_catalogue::mechanism_Ca_LVAst_gpu_advance_state_;
    result.write_ions=arb::bbp_catalogue::mechanism_Ca_LVAst_gpu_write_ions_;
    result.post_event=arb::bbp_catalogue::mechanism_Ca_LVAst_gpu_post_event_;
    return &result;
  }
};

