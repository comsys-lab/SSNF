`timescale 1ns/1ps
// XAPP052 taps 64,63,61,60 in right-shifting Galois form.
// Host can read the input bank in OFF mode; it remains architecturally observable.
module input_source #(parameter integer STATIC_FORMAT=-1)(
 input wire clk,rst,step,
 input wire [1:0] format_i,
 input wire seed_write,
 input wire [3:0] seed_word,
 input wire [31:0] seed_data,
 input wire [3:0] seed_strobe,
 input wire [4:0] read_word,
 output reg [1023:0] record_o,
 output reg [31:0] state_read,
 output wire [31:0] input_read,
 output wire bad_seed
);
 (* ram_style="registers" *) reg [63:0] state[0:5];
 wire [191:0] a={state[2],state[1],state[0]};
 wire [191:0] b={state[5],state[4],state[3]};
 assign bad_seed=(state[0]==0)||(state[1]==0)||(state[2]==0)||
                 (state[3]==0)||(state[4]==0)||(state[5]==0);
 assign input_read=record_o[read_word*32+:32];
 always @* begin
  state_read=0;
  for(integer j=0;j<12;j=j+1)
   if(read_word==j) state_read=state[j/2][(j%2)*32+:32];
 end
 function automatic [9:0] normalize7(input [6:0] magnitude);
  reg [7:0] x,y,z,w;
  reg a,b,c;
  reg [2:0] msb;
  begin
   x={1'b0,magnitude};
   a=(x[7:4]==0);y=a ? {x[3:0],4'b0} : x;
   b=(y[7:6]==0);z=b ? {y[5:0],2'b0} : y;
   c=(z[7]==0);w=c ? {z[6:0],1'b0} : z;
   msb=~{a,b,c};
   normalize7={msb,w[6:0]};
  end
 endfunction
 function automatic [7:0] exponent_lookup(input [5:0] key);
  begin
   case(key)
    6'd0:exponent_lookup=8'd127;
    6'd1:exponent_lookup=8'd128;
    6'd2:exponent_lookup=8'd129;
    6'd3:exponent_lookup=8'd130;
    6'd4:exponent_lookup=8'd131;
    6'd5:exponent_lookup=8'd132;
    6'd6:exponent_lookup=8'd133;
    6'd7:exponent_lookup=8'd134;
    6'd8:exponent_lookup=8'd128;
    6'd9:exponent_lookup=8'd129;
    6'd10:exponent_lookup=8'd130;
    6'd11:exponent_lookup=8'd131;
    6'd12:exponent_lookup=8'd132;
    6'd13:exponent_lookup=8'd133;
    6'd14:exponent_lookup=8'd134;
    6'd15:exponent_lookup=8'd135;
    6'd16:exponent_lookup=8'd129;
    6'd17:exponent_lookup=8'd130;
    6'd18:exponent_lookup=8'd131;
    6'd19:exponent_lookup=8'd132;
    6'd20:exponent_lookup=8'd133;
    6'd21:exponent_lookup=8'd134;
    6'd22:exponent_lookup=8'd135;
    6'd23:exponent_lookup=8'd136;
    6'd24:exponent_lookup=8'd130;
    6'd25:exponent_lookup=8'd131;
    6'd26:exponent_lookup=8'd132;
    6'd27:exponent_lookup=8'd133;
    6'd28:exponent_lookup=8'd134;
    6'd29:exponent_lookup=8'd135;
    6'd30:exponent_lookup=8'd136;
    6'd31:exponent_lookup=8'd137;
    6'd32:exponent_lookup=8'd123;
    6'd33:exponent_lookup=8'd124;
    6'd34:exponent_lookup=8'd125;
    6'd35:exponent_lookup=8'd126;
    6'd36:exponent_lookup=8'd127;
    6'd37:exponent_lookup=8'd128;
    6'd38:exponent_lookup=8'd129;
    6'd39:exponent_lookup=8'd130;
    6'd40:exponent_lookup=8'd124;
    6'd41:exponent_lookup=8'd125;
    6'd42:exponent_lookup=8'd126;
    6'd43:exponent_lookup=8'd127;
    6'd44:exponent_lookup=8'd128;
    6'd45:exponent_lookup=8'd129;
    6'd46:exponent_lookup=8'd130;
    6'd47:exponent_lookup=8'd131;
    6'd48:exponent_lookup=8'd125;
    6'd49:exponent_lookup=8'd126;
    6'd50:exponent_lookup=8'd127;
    6'd51:exponent_lookup=8'd128;
    6'd52:exponent_lookup=8'd129;
    6'd53:exponent_lookup=8'd130;
    6'd54:exponent_lookup=8'd131;
    6'd55:exponent_lookup=8'd132;
    6'd56:exponent_lookup=8'd126;
    6'd57:exponent_lookup=8'd127;
    6'd58:exponent_lookup=8'd128;
    6'd59:exponent_lookup=8'd129;
    6'd60:exponent_lookup=8'd130;
    6'd61:exponent_lookup=8'd131;
    6'd62:exponent_lookup=8'd132;
    6'd63:exponent_lookup=8'd133;
    default:exponent_lookup=0;
   endcase
  end
 endfunction
 function automatic [15:0] bf_encode(input signed [7:0] q,input integer e);
  reg [7:0] mag,ex;
  reg [9:0] normalized;
  begin
   mag=q[7] ? -q : q;
   normalized=normalize7(mag[6:0]);
   ex=exponent_lookup({e[2:0],normalized[9:7]});
   bf_encode=(mag==0)?16'd0:{q[7],ex,normalized[6:0]};
  end
 endfunction
 function automatic [511:0] encode(input [191:0] raw,input [1:0] fmt);
  integer e,h,l,q;
  reg [511:0] result;
  reg [7:0] scale;
  begin
   result=0;e=$signed(raw[172+:3]);scale=e+127;
   if(fmt==0) result[7:0]=scale;
   if(fmt==1) result[7:0]=scale+8'd6;
   for(integer i=0;i<32;i=i+1) begin
    h=$signed(raw[i*5+:5]);if(h==-16) h=-15;
    l=$signed(raw[160+(i/8)*3+:3]);q=8*h+l;
    case(fmt)
     0:result[8+i*5+:5]=h;
     1:result[8+i*8+:8]=q;
     2:result[i*16+:16]=bf_encode(q,e);
     default:begin end
    endcase
   end
   if(fmt==0) result[168+:12]=raw[160+:12];
   encode=result;
  end
 endfunction
 wire [1:0] selected_format=(STATIC_FORMAT<0)?format_i:STATIC_FORMAT;
 always @(posedge clk) begin
  if(rst) begin
   for(integer j=0;j<6;j=j+1) state[j]<=0;
   record_o<=0;
  end else begin
   if(seed_write) begin
    for(integer w=0;w<12;w=w+1) if(seed_word==w)
     for(integer byte_i=0;byte_i<4;byte_i=byte_i+1)
      if(seed_strobe[byte_i]) state[w/2][(w%2)*32+byte_i*8+:8]<=seed_data[byte_i*8+:8];
   end
   if(step) begin
    record_o<={encode(b,selected_format),encode(a,selected_format)};
    for(integer j=0;j<6;j=j+1)
     state[j]<=(state[j]>>1) ^ (state[j][0]?64'hd800000000000000:64'd0);
   end
  end
 end
endmodule
