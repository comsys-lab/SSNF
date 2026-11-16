if {$argc < 3 || $argc > 4} {error "ROOT OUTPUT PART [FORMAT] required"}
lassign $argv root output part format
file mkdir $output
read_verilog -sv [file join $root rtl input_source.sv]
if {$argc == 4} {
 synth_design -top input_source -part $part -mode out_of_context -generic STATIC_FORMAT=$format
} else {
 synth_design -top input_source -part $part -mode out_of_context
}
create_clock -period 10.0 [get_ports clk]
report_utilization -file [file join $output utilization.rpt]
report_timing_summary -file [file join $output timing.rpt]
write_checkpoint [file join $output source.dcp]
