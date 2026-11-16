# Read-only post-route review: CHECKPOINT OUTPUT_DIR
if {$argc != 2} {error "Expected routed checkpoint and output directory"}
lassign $argv checkpoint output
file mkdir $output
open_checkpoint $checkpoint
# Preserve the vendor clock relationships. A materially different image clock
# requires a matching checkpoint, not an invented generated-clock ratio.
set image [file join [file dirname $output] ssnf_bench.xclbin]
set topology [file join $output clock_topology.json]
exec xclbinutil --input $image --dump-section CLOCK_FREQ_TOPOLOGY:JSON:$topology
set runtime_mhz [exec python3 -c {import json,sys; d=json.load(open(sys.argv[1])); rows=d['clock_freq_topology']['m_clock_freq']; r=[x for x in rows if x['m_type']=='DATA']; assert len(r)==1; print(float(r[0]['m_freq_Mhz']))} $topology]
if {![string is double -strict $runtime_mhz] || $runtime_mhz <= 0 || $runtime_mhz > 200} {error "Invalid runtime DATA clock"}
set kernel_pin [get_pins level0_i/level1/level1_i/ulp/ss_ucs/aclk_kernel_00]
set kernel_clock [get_clocks -of_objects $kernel_pin]
if {[llength $kernel_clock] != 1} {error "Expected one kernel clock"}
set old_period [get_property PERIOD $kernel_clock]
set source_pin [get_property SOURCE $kernel_clock]
set target_pin [get_property SOURCE_PINS $kernel_clock]
if {[llength $target_pin] != 1 || [llength $source_pin] != 1} {error "Ambiguous generated clock pins"}
set source_clock [get_clocks -of_objects [get_pins $source_pin]]
if {[llength $source_clock] != 1} {error "Ambiguous generated clock source"}
set source_period [get_property PERIOD $source_clock]
set clock_name [get_property NAME $kernel_clock]
set reviewed_mhz [expr {1000.0 / $old_period}]
# This platform restores 9.999 ns for the nominal 10 ns clock. Accept at most
# 2 ps of reporting quantization, only in the stricter (faster) direction.
set period_difference [expr {1000.0 / $runtime_mhz - $old_period}]
if {$period_difference < -1e-9 || $period_difference > 0.002} {error "Checkpoint clock constraint does not match image clock"}
set clock_report [open [file join $output kernel_clock_properties.rpt] w]
puts $clock_report "Clock: $clock_name"
puts $clock_report "Checkpoint period ns: $old_period"
puts $clock_report "Source pin: $source_pin"
puts $clock_report "Target pin: $target_pin"
puts $clock_report "Source period ns: $source_period"
puts $clock_report "Xclbin runtime MHz: $runtime_mhz"
puts $clock_report "Reviewed MHz: $reviewed_mhz"
close $clock_report
report_timing_summary -check_timing_verbose -file [file join $output timing_routed.rpt]
report_bus_skew -file [file join $output bus_skew_routed.rpt]
report_utilization -hierarchical -file [file join $output utilization_routed.rpt]
report_drc -file [file join $output drc_routed.rpt]
report_clocks -file [file join $output clocks_routed.rpt]
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup_paths] || ![llength $hold_paths]} {error "No setup or hold paths found"}
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
set error_count [llength [get_drc_violations -filter {SEVERITY == Error}]]
set critical_count [llength [get_drc_violations -filter {SEVERITY == "Critical Warning"}]]
set hdpr5 [get_drc_violations -filter {NAME =~ HDPR-5*}]
set hdpr5_count [llength $hdpr5]
set f [open [file join $output drc_violations.txt] w]
foreach violation [get_drc_violations] {
  puts $f "$violation | [get_property SEVERITY $violation]"
}
close $f
set f [open [file join $output route_check.json] w]
puts $f "\{\"setup_slack_ns\":$setup_slack,\"hold_slack_ns\":$hold_slack,\"drc_error_count\":$error_count,\"drc_critical_warning_count\":$critical_count,\"hdpr5_violation_count\":$hdpr5_count,\"runtime_clock_mhz\":$runtime_mhz,\"reviewed_clock_mhz\":$reviewed_mhz,\"checkpoint_clock_mhz\":$reviewed_mhz,\"clock_period_difference_ns\":$period_difference,\"clock_constraints_preserved\":true\}"
close $f
set hierarchy_file [open [file join $output hierarchy_preservation.txt] w]
set f [open [file join $output core_inventory.txt] w]
foreach type {ssnf_dot32 mxint8_dot32 bf16_dot32 input_source} {
  set cells [get_cells -hier -filter "REF_NAME =~ *${type}* || ORIG_REF_NAME == $type"]
  puts $f "$type [llength $cells]"
  foreach cell $cells {
    puts $f "  $cell"
    set preserved [get_property KEEP_HIERARCHY $cell]
    puts $hierarchy_file "$cell | $preserved"
    if {[string tolower $preserved] ni {yes true 1}} {error "Source/arithmetic hierarchy was not preserved: $cell"}
  }
  if {[llength $cells] > 0} {
    report_utilization -cells [lindex [lsort $cells] 0] -file [file join $output ${type}_one_core_routed.rpt]
  }
}
close $f
close $hierarchy_file
close_design
