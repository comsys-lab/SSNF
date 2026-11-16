if {$argc != 2} {error "CHECKPOINT OUTPUT required"}
lassign $argv checkpoint output
open_checkpoint $checkpoint
set f [open $output w]
foreach type {input_source ssnf_dot32 mxint8_dot32 bf16_dot32} {
 set cells [get_cells -hier -filter "REF_NAME =~ *${type}* || ORIG_REF_NAME == $type"]
 foreach cell $cells {
  set v [get_property KEEP_HIERARCHY $cell]
  puts $f "$cell | $v"
  if {[string tolower $v] ni {yes true 1}} {error "Missing preserved boundary at $cell"}
 }
}
close $f
close_design
