`timescale 1ns/1ps
module bf16_mul(
  input wire clk,rst,ce,valid_i,
  input wire [15:0] a_i,b_i,
  output reg valid_o,
  output reg [31:0] result_o
);
  reg valid_q,sign_q,special_q;
  reg [31:0] special_value_q;
  reg [15:0] product_q;
  reg signed [10:0] exponent_q;
  wire az=(a_i[14:7]==0),bz=(b_i[14:7]==0);
  wire ai=(a_i[14:7]==255),bi=(b_i[14:7]==255);
  wire an=ai&&(|a_i[6:0]),bn=bi&&(|b_i[6:0]);
  integer out_exponent;
  reg [22:0] fraction;
  always @* begin
    out_exponent=$signed(exponent_q)+(product_q[15]?1:0);
    fraction=product_q[15]?{product_q[14:0],8'd0}:{product_q[13:0],9'd0};
  end
  always @(posedge clk) begin
    if(rst) begin valid_q<=0; valid_o<=0; end
    else if(ce) begin
      valid_q<=valid_i; valid_o<=valid_q;
      if(valid_i) begin
        sign_q<=a_i[15]^b_i[15];
        exponent_q<=$signed({1'b0,a_i[14:7]})+$signed({1'b0,b_i[14:7]})-11'sd127;
        product_q<={1'b1,a_i[6:0]}*{1'b1,b_i[6:0]};
        special_q<=az||bz||ai||bi;
        if(an||bn||(ai&&bz)||(bi&&az)) special_value_q<=32'h7fc00000;
        else if(ai||bi) special_value_q<={a_i[15]^b_i[15],8'hff,23'd0};
        else special_value_q<={a_i[15]^b_i[15],31'd0};
      end
      if(valid_q) begin
        if(special_q) result_o<=special_value_q;
        else if(out_exponent<=0) result_o<={sign_q,31'd0};
        else if(out_exponent>=255) result_o<={sign_q,8'hff,23'd0};
        else result_o<={sign_q,out_exponent[7:0],fraction};
      end
    end
  end
endmodule
