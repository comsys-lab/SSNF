`timescale 1ns/1ps
module mxint8_dot32(
  input wire clk,rst,ce,valid_i,
  input wire [1023:0] record_i,
  output reg valid_o,
  output reg [31:0] result_o,
  output reg signed [31:0] integer_o
);
  import fp_helpers::*;
  reg signed [15:0] products[0:31];
  wire [511:0] flat;
  wire signed [20:0] sum;
  wire sum_valid;
  reg [5:0] valids;
  reg signed [15:0] exponents[0:5];
  integer j;
  for(genvar i=0;i<32;i=i+1) begin: mul_g
    wire signed [7:0] a=record_i[8+i*8+:8];
    wire signed [7:0] b=record_i[520+i*8+:8];
    always @(posedge clk) if(ce&&valid_i) products[i]<=a*b;
    assign flat[i*16+:16]=products[i];
  end
  int_sum_tree #(.N(32),.W(16)) tree(clk,rst,ce,valids[0],flat,sum_valid,sum);
  wire signed [31:0] sum32=sum;
  always @(posedge clk) begin
    if(rst) begin valids<=0; valid_o<=0; end
    else if(ce) begin
      valids<={valids[4:0],valid_i}; valid_o<=sum_valid;
      if(valid_i) exponents[0]<=$signed({1'b0,record_i[7:0]})+
        $signed({1'b0,record_i[519:512]})-16'sd266;
      for(j=1;j<6;j=j+1) if(valids[j-1]) exponents[j]<=exponents[j-1];
      if(sum_valid) begin result_o<=scaled_int_fp32(sum32,exponents[5]); integer_o<=sum32; end
    end
  end
endmodule
