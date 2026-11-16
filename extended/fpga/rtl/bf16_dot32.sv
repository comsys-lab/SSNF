`timescale 1ns/1ps
module bf16_dot32(
  input wire clk,rst,ce,valid_i,
  input wire [1023:0] record_i,
  output wire valid_o,
  output wire [31:0] result_o
);
  wire [31:0] values[0:5][0:31];
  wire [31:0] valids[0:5];
  for(genvar i=0;i<32;i=i+1) begin: mul_g
    bf16_mul mul(clk,rst,ce,valid_i,record_i[i*16+:16],record_i[512+i*16+:16],valids[0][i],values[0][i]);
  end
  for(genvar l=1;l<=5;l=l+1) begin: level_g
    for(genvar i=0;i<(32>>l);i=i+1) begin: add_g
      fp32_add add(clk,rst,ce,valids[l-1][2*i],values[l-1][2*i],values[l-1][2*i+1],valids[l][i],values[l][i]);
    end
  end
  assign valid_o=valids[5][0];
  assign result_o=values[5][0];
endmodule
