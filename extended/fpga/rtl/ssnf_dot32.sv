`timescale 1ns/1ps
module ssnf_dot32(
  input wire clk,rst,ce,valid_i,
  input wire [1023:0] record_i,
  output reg valid_o,
  output reg [31:0] result_o,
  output reg signed [31:0] integer_o
);
  import fp_helpers::*;
  wire signed [4:0] ha[0:31],hb[0:31];
  wire signed [2:0] la[0:3],lb[0:3];
  reg signed [9:0] uu_q[0:31];
  reg signed [7:0] ga_q[0:3],gb_q[0:3];
  reg signed [2:0] la_q[0:3],lb_q[0:3];
  reg signed [10:0] us_q[0:3],su_q[0:3];
  reg signed [5:0] ss_q[0:3];
  reg [6:0] valid_pipe;
  reg signed [15:0] exponent_pipe[0:6];
  wire [319:0] uu_flat;
  wire [43:0] us_flat,su_flat;
  wire [23:0] ss_flat;
  wire signed [14:0] uu_sum;
  wire signed [12:0] us_sum,su_sum;
  wire signed [7:0] ss_sum;
  wire vu,vus,vsu,vss;
  reg signed [12:0] us_delay[0:1],su_delay[0:1];
  reg signed [7:0] ss_delay[0:1];
  reg signed [31:0] final_q;
  integer k;
  for(genvar i=0;i<32;i=i+1) begin: upper_g
    assign ha[i]=$signed(record_i[8+i*5+:5]);
    assign hb[i]=$signed(record_i[520+i*5+:5]);
    assign uu_flat[i*10+:10]=uu_q[i];
    always @(posedge clk) if(ce&&valid_i) uu_q[i]<=ha[i]*hb[i];
  end
  for(genvar g=0;g<4;g=g+1) begin: group_g
    wire signed [7:0] a[0:7],b[0:7];
    wire signed [7:0] as0,as1,as2,as3,bs0,bs1,bs2,bs3;
    for(genvar j=0;j<8;j=j+1) begin: extend_g
      assign a[j]=ha[g*8+j]; assign b[j]=hb[g*8+j];
    end
    assign as0=a[0]+a[1]; assign as1=a[2]+a[3];
    assign as2=a[4]+a[5]; assign as3=a[6]+a[7];
    assign bs0=b[0]+b[1]; assign bs1=b[2]+b[3];
    assign bs2=b[4]+b[5]; assign bs3=b[6]+b[7];
    assign la[g]=$signed(record_i[168+g*3+:3]);
    assign lb[g]=$signed(record_i[680+g*3+:3]);
    assign us_flat[g*11+:11]=us_q[g];
    assign su_flat[g*11+:11]=su_q[g];
    assign ss_flat[g*6+:6]=ss_q[g];
    always @(posedge clk) if(ce) begin
      if(valid_i) begin
        ga_q[g]<=(as0+as1)+(as2+as3); gb_q[g]<=(bs0+bs1)+(bs2+bs3);
        la_q[g]<=la[g]; lb_q[g]<=lb[g];
      end
      if(valid_pipe[0]) begin
        us_q[g]<=ga_q[g]*lb_q[g]; su_q[g]<=gb_q[g]*la_q[g];
        ss_q[g]<=la_q[g]*lb_q[g];
      end
    end
  end
  int_sum_tree #(.N(32),.W(10)) tu(clk,rst,ce,valid_pipe[0],uu_flat,vu,uu_sum);
  int_sum_tree #(.N(4),.W(11)) tus(clk,rst,ce,valid_pipe[1],us_flat,vus,us_sum);
  int_sum_tree #(.N(4),.W(11)) tsu(clk,rst,ce,valid_pipe[1],su_flat,vsu,su_sum);
  int_sum_tree #(.N(4),.W(6)) tss(clk,rst,ce,valid_pipe[1],ss_flat,vss,ss_sum);
  wire signed [31:0] uu32=uu_sum,us32=us_delay[1],su32=su_delay[1],ss32=ss_delay[1];
  always @(posedge clk) begin
    if(rst) begin valid_pipe<=0; valid_o<=0; end
    else if(ce) begin
      valid_pipe<={valid_pipe[5:0],valid_i}; valid_o<=valid_pipe[6];
      if(valid_i) exponent_pipe[0]<= $signed({1'b0,record_i[7:0]})+
          $signed({1'b0,record_i[519:512]})-16'sd254;
      for(k=1;k<7;k=k+1) if(valid_pipe[k-1]) exponent_pipe[k]<=exponent_pipe[k-1];
      if(vus) begin us_delay[0]<=us_sum; su_delay[0]<=su_sum; ss_delay[0]<=ss_sum; end
      if(valid_pipe[4]) begin
        us_delay[1]<=us_delay[0]; su_delay[1]<=su_delay[0]; ss_delay[1]<=ss_delay[0];
      end
      if(vu) final_q<=(uu32<<<6)+(us32<<<3)+(su32<<<3)+(ss32<<<3);
      if(valid_pipe[6]) begin
        result_o<=scaled_int_fp32(final_q,exponent_pipe[6]); integer_o<=final_q;
      end
    end
  end
endmodule
