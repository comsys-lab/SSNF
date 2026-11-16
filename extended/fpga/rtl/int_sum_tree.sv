`timescale 1ns/1ps
module int_sum_tree #(parameter integer N=32, parameter integer W=32)(
  input wire clk,rst,ce,valid_i,
  input wire [N*W-1:0] values_i,
  output wire valid_o,
  output wire signed [W+$clog2(N)-1:0] value_o
);
  localparam integer LEVELS=$clog2(N);
  wire signed [W+LEVELS-1:0] nodes[0:LEVELS][0:N-1];
  wire [LEVELS:0] valids;
  assign valids[0]=valid_i;
  for(genvar i=0;i<N;i=i+1) begin: inputs_g
    assign nodes[0][i]=$signed(values_i[i*W+:W]);
  end
  for(genvar l=1;l<=LEVELS;l=l+1) begin: levels_g
    reg valid_q;
    always @(posedge clk) begin
      if(rst) valid_q<=0;
      else if(ce) valid_q<=valids[l-1];
    end
    assign valids[l]=valid_q;
    for(genvar i=0;i<(N>>l);i=i+1) begin: add_g
      reg signed [W+LEVELS-1:0] sum_q;
      always @(posedge clk) if(ce&&valids[l-1])
        sum_q<=nodes[l-1][2*i]+nodes[l-1][2*i+1];
      assign nodes[l][i]=sum_q;
    end
  end
  assign valid_o=valids[LEVELS];
  assign value_o=nodes[LEVELS][0];
endmodule
