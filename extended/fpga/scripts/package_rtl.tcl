# vivado -mode batch -source package_rtl.tcl -tclargs ROOT BUILD PART REPLICAS FORMAT
if {$argc != 5} {error "Expected ROOT BUILD PART REPLICAS FORMAT"}
lassign $argv root build part replicas format
file mkdir $build
create_project package_ssnf [file join $build package_project] -part $part
set sources [list fp_helpers.sv int_sum_tree.sv ssnf_dot32.sv mxint8_dot32.sv bf16_mul.sv fp32_add.sv bf16_dot32.sv input_source.sv ssnf_bench.sv]
foreach source $sources {add_files [file join $root rtl $source]}
set_property top ssnf_bench [current_fileset]
set_property generic "MAX_REPLICAS=$replicas STATIC_FORMAT=$format" [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir [file join $build ip] -vendor ssnf.local -library rtl -taxonomy /KernelIP -import_files
set core [ipx::current_core]
set_property name ssnf_bench $core
set_property version "1.[expr {$format+1}]" $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property vitis_drc {ctrl_protocol user_managed} $core
set_property ipi_drc {ignore_freq_hz true} $core
set p [ipx::get_user_parameters MAX_REPLICAS -of_objects $core]
if {[llength $p]} {set_property value $replicas $p}
set hdl_p [ipx::get_hdl_parameters MAX_REPLICAS -of_objects $core]
if {[llength $hdl_p] != 1} {error "Missing MAX_REPLICAS HDL parameter"}
set_property value $replicas $hdl_p
set format_user [ipx::get_user_parameters STATIC_FORMAT -of_objects $core]
set format_hdl [ipx::get_hdl_parameters STATIC_FORMAT -of_objects $core]
if {[llength $format_user]!=1 || [llength $format_hdl]!=1} {error "Missing static format parameter"}
set_property value $format $format_user
set_property value $format $format_hdl
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core
set mm [ipx::get_memory_maps s_axi_control -of_objects $core]
if {![llength $mm]} {set mm [ipx::add_memory_map s_axi_control $core]}
set blocks [ipx::get_address_blocks -of_objects $mm]
if {![llength $blocks]} {set block [ipx::add_address_block reg0 $mm]} else {set block [lindex $blocks 0]}
set_property range 65536 $block
set_property width 32 $block
set_property usage register $block
set bus [ipx::get_bus_interfaces s_axi_control -of_objects $core]
set_property slave_memory_map_ref s_axi_control $bus
foreach {name offset} {USER_CTRL 0 MODE 4 REPLICAS 8 FLAGS 12 LIMIT_LO 16 LIMIT_HI 20 INTERVAL 24 STATUS 28 EVENTS_LO 32 EVENTS_HI 36 CYCLES_LO 40 CYCLES_HI 44 BATCHES_LO 48 BATCHES_HI 52 FIRST_CYCLE 56 ABI 60 ACCEPTED_LO 64 ACCEPTED_HI 68 MAX_REPLICAS_REG 72 STATIC_FORMAT_REG 76} {
  set reg [ipx::add_register $name $block]
  set_property address_offset $offset $reg
  set_property size 32 $reg
}
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
exec python3 -c {import sys,xml.etree.ElementTree as E; r=E.parse(sys.argv[1]).getroot(); ns={'s':'http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009'}; p=[x for x in r.findall('.//s:modelParameter',ns) if x.find('s:name',ns).text=='MAX_REPLICAS']; assert len(p)==1 and int(p[0].find('s:value',ns).text)==int(sys.argv[2]),'HDL replica parameter mismatch'} [file join $build ip component.xml] $replicas
exec python3 -c {import sys,xml.etree.ElementTree as E;ns={'s':'http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009'};r=E.parse(sys.argv[1]).getroot();p=[x for x in r.findall('.//s:modelParameter',ns) if x.find('s:name',ns).text=='STATIC_FORMAT'];assert len(p)==1 and int(p[0].find('s:value',ns).text)==int(sys.argv[2])} [file join $build ip component.xml] $format
package_xo -xo_path [file join $build ssnf_bench.xo] -kernel_name ssnf_bench -ip_directory [file join $build ip] -ctrl_protocol user_managed
close_project
