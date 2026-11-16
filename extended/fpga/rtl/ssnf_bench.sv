`timescale 1ns/1ps
module ssnf_bench #(parameter integer MAX_REPLICAS=16,parameter integer STATIC_FORMAT=-1)(
  input wire ap_clk,ap_rst_n,
  input wire [15:0] s_axi_control_awaddr,
  input wire s_axi_control_awvalid,
  output wire s_axi_control_awready,
  input wire [31:0] s_axi_control_wdata,
  input wire [3:0] s_axi_control_wstrb,
  input wire s_axi_control_wvalid,
  output wire s_axi_control_wready,
  output reg [1:0] s_axi_control_bresp,
  output reg s_axi_control_bvalid,
  input wire s_axi_control_bready,
  input wire [15:0] s_axi_control_araddr,
  input wire s_axi_control_arvalid,
  output wire s_axi_control_arready,
  output reg [31:0] s_axi_control_rdata,
  output reg [1:0] s_axi_control_rresp,
  output reg s_axi_control_rvalid,
  input wire s_axi_control_rready
);
  wire rst=!ap_rst_n;
  reg aw_pending,w_pending;
  reg [15:0] write_addr;
  reg [31:0] write_data;
  reg [3:0] write_strobe;
  assign s_axi_control_awready=!aw_pending&&!s_axi_control_bvalid;
  assign s_axi_control_wready=!w_pending&&!s_axi_control_bvalid;
  assign s_axi_control_arready=!s_axi_control_rvalid;
  wire write_fire=aw_pending&&w_pending&&!s_axi_control_bvalid;
  wire control_write=write_fire&&(write_addr==16'h0000)&&write_strobe[0];
  wire start_command=control_write&&write_data[0];
  wire stop_command=control_write&&write_data[1];
  wire snapshot_command=control_write&&write_data[2];
  wire clear_command=control_write&&write_data[3];


  reg [31:0] cfg_mode,cfg_replicas,cfg_flags,cfg_interval;
  reg [63:0] cfg_limit;
  reg busy,done,error_q,stop_q,bank_valid;
  reg [31:0] cooldown;
  reg [63:0] issued_batches,accepted_count,event_count,active_cycles;
  reg [63:0] snap_accepted,snap_events,snap_cycles,snap_batches;
  reg [31:0] first_latency;
  reg [16:0] expected_pipe;
  wire expected_valid=(cfg_mode==0)?expected_pipe[7]:((cfg_mode==1)?expected_pipe[6]:expected_pipe[16]);
  wire [MAX_REPLICAS-1:0] bad_seeds,actual_valid;
  wire [31:0] results[0:MAX_REPLICAS-1];
  wire [31:0] state_reads[0:MAX_REPLICAS-1],input_reads[0:MAX_REPLICAS-1];
  reg any_bad;
  always @* begin
    any_bad=0;
    for(integer j=0;j<MAX_REPLICAS;j=j+1) if(j<cfg_replicas) any_bad=any_bad|bad_seeds[j];
  end
  wire config_valid=(STATIC_FORMAT<0||cfg_mode==STATIC_FORMAT)&&(cfg_mode<3)&&(cfg_replicas>=1)&&(cfg_replicas<=MAX_REPLICAS)&&
                    (cfg_interval>=1)&&(!any_bad||cfg_flags[1]);
  wire start_ok=start_command&&!busy&&config_valid;
  wire no_more_inputs=stop_q||stop_command||((cfg_limit!=0)&&(issued_batches>=cfg_limit));
  wire issue=busy&&!no_more_inputs&&(cooldown==0);
  wire core_reset=rst||start_ok;
  wire [4:0] source_read_word=(s_axi_control_araddr[15:12]==1)?{1'b0,s_axi_control_araddr[5:2]}:s_axi_control_araddr[6:2];
  for(genvar r=0;r<MAX_REPLICAS;r=r+1) begin: replica_g
    localparam integer RID=r;
    wire active=busy&&(RID<cfg_replicas);
    wire source_step=issue&&(RID<cfg_replicas);
    wire computing=active&&cfg_flags[0];
    wire seed_write=write_fire&&!busy&&(write_addr[15:12]==1)&&(write_addr[11:8]==r);
    wire [1023:0] record;
    wire [1023:0] ssnf_record=(computing&&(cfg_mode==0))?record:1024'd0;
    wire [1023:0] mx_record=(computing&&(cfg_mode==1))?record:1024'd0;
    wire [1023:0] bf_record=(computing&&(cfg_mode==2))?record:1024'd0;
    (* keep_hierarchy="yes" *) input_source #(.STATIC_FORMAT(STATIC_FORMAT)) source(ap_clk,rst,source_step,cfg_mode[1:0],seed_write,
      write_addr[5:2],write_data,write_strobe,source_read_word,record,state_reads[r],input_reads[r],bad_seeds[r]);
    wire vs,vm,vb;
    wire [31:0] ys,ym,yb,unused_s,unused_m;
    if(STATIC_FORMAT<0||STATIC_FORMAT==0) begin: ssnf_g
    (* keep_hierarchy="yes" *) ssnf_dot32 ssnf(ap_clk,core_reset,computing&&(cfg_mode==0),bank_valid&&(cfg_mode==0),ssnf_record,vs,ys,unused_s);
    end else begin
     assign vs=0;assign ys=0;assign unused_s=0;
    end
    if(STATIC_FORMAT<0||STATIC_FORMAT==1) begin: mx_g
    (* keep_hierarchy="yes" *) mxint8_dot32 mx(ap_clk,core_reset,computing&&(cfg_mode==1),bank_valid&&(cfg_mode==1),mx_record,vm,ym,unused_m);
    end else begin
     assign vm=0;assign ym=0;assign unused_m=0;
    end
    if(STATIC_FORMAT<0||STATIC_FORMAT==2) begin: bf_g
    (* keep_hierarchy="yes" *) bf16_dot32 bf(ap_clk,core_reset,computing&&(cfg_mode==2),bank_valid&&(cfg_mode==2),bf_record,vb,yb);
    end else begin
     assign vb=0;assign yb=0;
    end
    assign actual_valid[r]=computing&&((cfg_mode==0)?vs:((cfg_mode==1)?vm:vb));
    assign results[r]=(cfg_mode==0)?ys:((cfg_mode==1)?ym:yb);
  end
  integer events_now;
  always @* begin
    events_now=0;
    if(cfg_flags[0]) begin
      for(integer j=0;j<MAX_REPLICAS;j=j+1) if(actual_valid[j]) events_now=events_now+1;
    end else if(expected_valid) events_now=cfg_replicas;
  end
  function automatic [31:0] merge32(input [31:0] old_value,input [31:0] new_value,input [3:0] strobe);
    begin
      merge32=old_value;
      for(integer b=0;b<4;b=b+1) if(strobe[b]) merge32[b*8+:8]=new_value[b*8+:8];
    end
  endfunction



  function automatic [31:0] read_register(input [15:0] addr);
    begin
      case(addr)
        0:read_register=0;
        4:read_register=cfg_mode;
        8:read_register=cfg_replicas;
        12:read_register=cfg_flags;
        16:read_register=cfg_limit[31:0];
        20:read_register=cfg_limit[63:32];
        24:read_register=cfg_interval;
        28:read_register={29'd0,error_q,done,busy};
        32:read_register=snap_events[31:0];
        36:read_register=snap_events[63:32];
        40:read_register=snap_cycles[31:0];
        44:read_register=snap_cycles[63:32];
        48:read_register=snap_batches[31:0];
        52:read_register=snap_batches[63:32];
        56:read_register=first_latency;
        60:read_register=32'h53534e32;
        64:read_register=snap_accepted[31:0];
        68:read_register=snap_accepted[63:32];
        72:read_register=MAX_REPLICAS;
        76:read_register=STATIC_FORMAT;
        default:begin
          read_register=0;
          for(integer j=0;j<MAX_REPLICAS;j=j+1) begin
            if(addr==16'h0800+j*4) read_register=results[j];
            if((addr[15:12]==1)&&(addr[11:8]==j)&&(addr[7:2]<12)) read_register=state_reads[j];
            if((addr[15:12]==4)&&(addr[11:8]==j)&&(addr[7]==0)) read_register=input_reads[j];
          end
        end
      endcase
    end
  endfunction

  always @(posedge ap_clk) begin
    if(rst) begin
      aw_pending<=0; w_pending<=0; s_axi_control_bvalid<=0; s_axi_control_rvalid<=0;
      s_axi_control_bresp<=0; s_axi_control_rresp<=0;
    end else begin
      if(s_axi_control_awready&&s_axi_control_awvalid) begin
        write_addr<=s_axi_control_awaddr; aw_pending<=1;
      end
      if(s_axi_control_wready&&s_axi_control_wvalid) begin
        write_data<=s_axi_control_wdata; write_strobe<=s_axi_control_wstrb; w_pending<=1;
      end
      if(s_axi_control_bvalid&&s_axi_control_bready) s_axi_control_bvalid<=0;
      if(write_fire) begin
        aw_pending<=0;w_pending<=0;s_axi_control_bvalid<=1;
        s_axi_control_bresp<=(busy&&(write_addr!=0))?2'b10:2'b00;
      end
      if(s_axi_control_rvalid&&s_axi_control_rready) s_axi_control_rvalid<=0;
      if(s_axi_control_arready&&s_axi_control_arvalid) begin
        s_axi_control_rdata<=read_register(s_axi_control_araddr);
        s_axi_control_rresp<=2'b00;
        s_axi_control_rvalid<=1;
      end
    end
  end



  always @(posedge ap_clk) begin
    if(rst) begin
      cfg_mode<=0;cfg_replicas<=1;cfg_flags<=1;cfg_interval<=1;cfg_limit<=1;
      busy<=0;done<=0;error_q<=0;stop_q<=0;bank_valid<=0;cooldown<=0;expected_pipe<=0;
      issued_batches<=0;accepted_count<=0;event_count<=0;active_cycles<=0;first_latency<=0;
      snap_events<=0;snap_cycles<=0;snap_accepted<=0;snap_batches<=0;
    end else begin
      bank_valid<=issue;
      expected_pipe<={expected_pipe[15:0],bank_valid};
      if(clear_command&&!busy) begin done<=0;error_q<=0;end
      if(write_fire&&busy&&(write_addr!=0)) error_q<=1;
      if(write_fire&&!busy) begin
        case(write_addr)
          4:cfg_mode<=merge32(cfg_mode,write_data,write_strobe);
          8:cfg_replicas<=merge32(cfg_replicas,write_data,write_strobe);
          12:cfg_flags<=merge32(cfg_flags,write_data,write_strobe);
          16:cfg_limit[31:0]<=merge32(cfg_limit[31:0],write_data,write_strobe);
          20:cfg_limit[63:32]<=merge32(cfg_limit[63:32],write_data,write_strobe);
          24:cfg_interval<=merge32(cfg_interval,write_data,write_strobe);
          default:begin end
        endcase
      end
      if(start_command) begin
        if(!start_ok) error_q<=1;
        else begin
          busy<=1;done<=0;error_q<=0;stop_q<=0;bank_valid<=0;cooldown<=0;expected_pipe<=0;
          issued_batches<=0;accepted_count<=0;event_count<=0;active_cycles<=0;first_latency<=0;
        end
      end
      if(busy) begin
        active_cycles<=active_cycles+1;
        if(stop_command) stop_q<=1;
        if(cooldown!=0) cooldown<=cooldown-1;
        if(issue) begin issued_batches<=issued_batches+1;cooldown<=cfg_interval-1;end
        if(bank_valid) accepted_count<=accepted_count+cfg_replicas;
        if(events_now!=0) begin
          event_count<=event_count+events_now;
          if(event_count==0) first_latency<=active_cycles;
        end
        if(cfg_flags[0]) begin
          for(integer j=0;j<MAX_REPLICAS;j=j+1)
            if(j<cfg_replicas && actual_valid[j]!=expected_valid) error_q<=1;
        end
        if(no_more_inputs&&!bank_valid&&(expected_pipe==0)&&(accepted_count==event_count)) begin busy<=0;done<=1;end
      end
      if(snapshot_command) begin
        snap_events<=event_count;snap_cycles<=active_cycles;
        snap_accepted<=accepted_count;snap_batches<=issued_batches;
      end
    end
  end

endmodule
