`timescale 1ns/1ps
module tb_source;
 reg clk=0;always #5 clk=~clk;
 reg rst=1,step=0,write_en=0;
 reg [1:0] mode=0;
 reg [3:0] word=0;
 reg [31:0] data=0;
 reg [4:0] rd=0;
 wire [1023:0] record;
 wire [31:0] sr,ir;
 wire bad;
 input_source dut(clk,rst,step,mode,write_en,word,data,4'hf,rd,record,sr,ir,bad);
 string filename;integer f,rc,m,count=0;
 reg [383:0] seeds,next_seeds;
 reg [1023:0] expected;
 initial begin
  if(!$value$plusargs("VECTORS=%s",filename)) $fatal(1,"vectors required");
  f=$fopen(filename,"r");if(!f)$fatal(1,"file missing");
  repeat(3) @(negedge clk);rst=0;
  while(!$feof(f)) begin
   rc=$fscanf(f,"%d %h %h %h\n",m,seeds,expected,next_seeds);
   if(rc==4) begin
    mode=m;
    for(integer j=0;j<12;j=j+1) begin
     @(negedge clk);write_en=1;word=j;data=seeds[j*32+:32];
    end
    @(negedge clk);write_en=0;step=1;
    @(negedge clk);step=0;
    if(record!==expected)$fatal(1,"Encoded source mismatch case=%0d mode=%0d",count,m);
    for(integer j=0;j<12;j=j+1) begin
     rd=j;#1;if(sr!==next_seeds[j*32+:32])$fatal(1,"Source state mismatch");
    end
    for(integer j=0;j<32;j=j+1) begin
     rd=j;#1;if(ir!==expected[j*32+:32])$fatal(1,"Input bank read mismatch");
    end
    count=count+1;
   end
  end
  $display("PASS SOURCE %0d records, encodings, state transitions and bank reads",count);$finish;
 end
 initial begin #100000000;$fatal(1,"timeout");end
endmodule
