# Offline synthesis; never programs a board. ROOT BUILD PART REPLICAS CLOCK_MHZ FORMAT
if {$argc != 6} {error "Expected ROOT BUILD PART REPLICAS CLOCK_MHZ FORMAT"}
lassign $argv root build part replicas mhz format
file mkdir $build
foreach source {fp_helpers.sv int_sum_tree.sv ssnf_dot32.sv mxint8_dot32.sv bf16_mul.sv fp32_add.sv bf16_dot32.sv input_source.sv ssnf_bench.sv} {
  read_verilog -sv [file join $root rtl $source]
}
synth_design -mode out_of_context -top ssnf_bench -part $part -generic [list MAX_REPLICAS=$replicas STATIC_FORMAT=$format]
create_clock -name ap_clk -period [expr {1000.0 / $mhz}] [get_ports ap_clk]
report_utilization -hierarchical -file [file join $build utilization_synth.rpt]
report_timing_summary -file [file join $build timing_synth.rpt]
write_checkpoint [file join $build synthesized.dcp]
