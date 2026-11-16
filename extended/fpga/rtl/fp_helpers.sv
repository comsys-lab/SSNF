`timescale 1ns/1ps
package fp_helpers;
  // Signed integer times a power of two -> FP32, RNE, flush subnormal results.
  function automatic [31:0] scaled_int_fp32(
      input logic signed [31:0] number,
      input logic signed [15:0] scale_exp);
    logic sign_bit;
    logic [31:0] mag;
    logic [63:0] wide, sig, rem_bits, halfway;
    integer msb, j, shift, exponent;
    begin
      sign_bit = number[31];
      mag = sign_bit ? (~number + 32'd1) : number;
      msb = 0;
      for (j=0;j<32;j=j+1) if (mag[j]) msb=j;
      wide = {32'd0,mag};
      if (msb>23) begin
        shift=msb-23;
        sig=wide>>shift;
        rem_bits=wide & ((64'd1<<shift)-1);
        halfway=64'd1<<(shift-1);
        if ((rem_bits>halfway)||((rem_bits==halfway)&&sig[0])) sig=sig+1;
        if (sig[24]) begin sig=sig>>1; msb=msb+1; end
      end else sig=wide<<(23-msb);
      exponent=msb+127+$signed(scale_exp);
      if (mag==0) scaled_int_fp32=32'd0;
      else if (exponent<=0) scaled_int_fp32={sign_bit,31'd0};
      else if (exponent>=255) scaled_int_fp32={sign_bit,8'hff,23'd0};
      else scaled_int_fp32={sign_bit,exponent[7:0],sig[22:0]};
    end
  endfunction

  function automatic [26:0] shr_jam27(input logic [26:0] value,input integer distance);
    logic sticky;
    integer k;
    begin
      sticky=0;
      if (distance>=27) shr_jam27={26'd0,|value};
      else if (distance<=0) shr_jam27=value;
      else begin
        for(k=0;k<27;k=k+1) if(k<distance) sticky=sticky|value[k];
        shr_jam27=(value>>distance)|{26'd0,sticky};
      end
    end
  endfunction
endpackage
