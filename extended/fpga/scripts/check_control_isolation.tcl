# Read-only verification of cross-replica source -> arithmetic control feedback.
# CHECKPOINT OUTPUT_DIR EXPECT_REMOVED (0 for the retained old diagnostic).
if {$argc != 3} {error "Expected CHECKPOINT OUTPUT_DIR EXPECT_REMOVED"}
lassign $argv checkpoint output expect_removed
file mkdir $output
open_checkpoint $checkpoint
set qs {}
foreach c [get_cells -hier -filter {NAME =~ *source/state_reg*}] {
  if {[string first {replica_g[7].source/} $c]>=0} {
    foreach p [get_pins -of_objects $c -filter {REF_PIN_NAME == Q}] {lappend qs [get_property NAME $p]}
  }
}
set controls {}
foreach p [get_pins -hier -filter {NAME =~ *bf_g.bf* && (REF_PIN_NAME == CE || REF_PIN_NAME == R)}] {
  if {[string first {replica_g[0].bf_g.bf/} $p]>=0} {lappend controls [get_property NAME $p]}
}
if {[llength $qs]<384 || ![llength $controls]} {error "Missing cross-replica diagnostic endpoints"}
set source_pins [get_pins $qs]
set control_pins [get_pins $controls]
set endpoint_file [open [file join $output endpoints.txt] w]
puts $endpoint_file "Source: [lindex $qs 0] ... [lindex $qs end]"
puts $endpoint_file "Destination: [lindex $controls 0] ... [lindex $controls end]"
close $endpoint_file
if {[llength $source_pins]!=[llength $qs] || [llength $control_pins]!=[llength $controls]} {error "Endpoint names did not resolve exactly"}
set paths [get_timing_paths -from $source_pins -to $control_pins -max_paths 3]
report_timing -from $source_pins -to $control_pins -max_paths 3 -file [file join $output cross_replica_control.rpt]
set f [open [file join $output control_isolation.json] w]
puts $f "\{\"source_q_count\":[llength $qs],\"control_pin_count\":[llength $controls],\"cross_replica_path_count_capped_at_3\":[llength $paths],\"expected_removed\":$expect_removed\}"
close $f
if {$expect_removed && [llength $paths]} {error "Streaming source still drives another replica's arithmetic control"}
if {!$expect_removed && ![llength $paths]} {error "Old diagnostic did not reproduce the control feedback"}
close_design
