`timescale 1ns/1ps
// Three actual stages: magnitude ordering/alignment; add/subtract; normalize/RNE.
// Finite normals and zero are the measurement domain. Subnormals are FTZ.
module fp32_add(
  input wire clk,rst,ce,valid_i,
  input wire [31:0] a_i,b_i,
  output reg valid_o,
  output reg [31:0] result_o
);
  import fp_helpers::*;
  reg v1,v2;
  reg [26:0] large_q,small_q;
  reg sign1,subtract_q,special1;
  reg [31:0] special_value1;
  reg [8:0] exponent1;
  reg [27:0] sum2;
  reg sign2,special2;
  reg [31:0] special_value2;
  reg [8:0] exponent2;
  reg [31:0] a,b,big_operand,small_operand;
  reg [26:0] normalized;
  reg [24:0] rounded;
  reg [31:0] packed_result;
  integer exponent_out,shift,j;
  always @* begin
    a=(a_i[30:23]==0)?{a_i[31],31'd0}:a_i;
    b=(b_i[30:23]==0)?{b_i[31],31'd0}:b_i;
    if(a[30:0]>=b[30:0]) begin big_operand=a; small_operand=b; end
    else begin big_operand=b; small_operand=a; end
  end
  always @* begin
    exponent_out=exponent2;
    if(sum2[27]) begin normalized=sum2[27:1]; normalized[0]=normalized[0]|sum2[0]; exponent_out=exponent_out+1; end
    else begin
      shift=0;
      for(j=0;j<27;j=j+1) if(sum2[j]) shift=26-j;
      normalized=sum2[26:0]<<shift; exponent_out=exponent_out-shift;
    end
    rounded={1'b0,normalized[26:3]};
    if(normalized[2]&&(normalized[1]||normalized[0]||normalized[3])) rounded=rounded+1;
    if(rounded[24]) begin rounded=rounded>>1; exponent_out=exponent_out+1; end
    if(sum2==0) packed_result=32'd0;
    else if(exponent_out<=0) packed_result={sign2,31'd0};
    else if(exponent_out>=255) packed_result={sign2,8'hff,23'd0};
    else packed_result={sign2,exponent_out[7:0],rounded[22:0]};
  end
  always @(posedge clk) begin
    if(rst) begin v1<=0; v2<=0; valid_o<=0; end
    else if(ce) begin
      v1<=valid_i; v2<=v1; valid_o<=v2;
      if(valid_i) begin
        large_q<={1'b1,big_operand[22:0],3'd0};
        small_q<=shr_jam27({1'b1,small_operand[22:0],3'd0},big_operand[30:23]-small_operand[30:23]);
        sign1<=big_operand[31]; subtract_q<=big_operand[31]^small_operand[31]; exponent1<={1'b0,big_operand[30:23]};
        special1<=((a[30:23]==255)||(b[30:23]==255)||(a[30:0]==0)||(b[30:0]==0));
        if(((a[30:23]==255)&&(|a[22:0]))||((b[30:23]==255)&&(|b[22:0]))||
           ((a[30:23]==255)&&(b[30:23]==255)&&(a[31]!=b[31]))) special_value1<=32'h7fc00000;
        else if(a[30:23]==255) special_value1<=a;
        else if(b[30:23]==255) special_value1<=b;
        else if((a[30:0]==0)&&(b[30:0]==0)) special_value1<={a[31]&b[31],31'd0};
        else if(a[30:0]==0) special_value1<=b;
        else special_value1<=a;
      end
      if(v1) begin
        sum2<=subtract_q?({1'b0,large_q}-{1'b0,small_q}):({1'b0,large_q}+{1'b0,small_q});
        exponent2<=exponent1; sign2<=sign1; special2<=special1; special_value2<=special_value1;
      end
      if(v2) result_o<=special2?special_value2:packed_result;
    end
  end
endmodule
