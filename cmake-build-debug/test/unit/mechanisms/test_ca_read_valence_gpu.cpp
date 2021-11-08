#include <arbor/mechanism_abi.h>
#include <cmath>

namespace testing {
void mechanism_test_ca_read_valence_gpu_init_(arb_mechanism_ppack*);
void mechanism_test_ca_read_valence_gpu_advance_state_(arb_mechanism_ppack*);
void mechanism_test_ca_read_valence_gpu_compute_currents_(arb_mechanism_ppack*);
void mechanism_test_ca_read_valence_gpu_write_ions_(arb_mechanism_ppack*);
void mechanism_test_ca_read_valence_gpu_apply_events_(arb_mechanism_ppack*, arb_deliverable_event_stream*);
void mechanism_test_ca_read_valence_gpu_post_event_(arb_mechanism_ppack*);

} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test_ca_read_valence_interface_gpu() {
    static arb_mechanism_interface result;
    result.backend=arb_backend_kind_gpu;
    result.partition_width=1;
    result.alignment=1;
    result.init_mechanism=testing::mechanism_test_ca_read_valence_gpu_init_;
    result.compute_currents=testing::mechanism_test_ca_read_valence_gpu_compute_currents_;
    result.apply_events=testing::mechanism_test_ca_read_valence_gpu_apply_events_;
    result.advance_state=testing::mechanism_test_ca_read_valence_gpu_advance_state_;
    result.write_ions=testing::mechanism_test_ca_read_valence_gpu_write_ions_;
    result.post_event=testing::mechanism_test_ca_read_valence_gpu_post_event_;
    return &result;
  }
};

