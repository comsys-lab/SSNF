if {$argc != 2} {error "CHECKPOINT OUTPUT required"}
lassign $argv checkpoint output
open_checkpoint $checkpoint
set f [open $output w]
foreach cell [get_cells -filter {IS_PRIMITIVE == 1}] {
 puts $f "[get_property NAME $cell]\t[get_property REF_NAME $cell]"
}
close $f
close_design
