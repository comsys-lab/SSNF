`timescale 1ns/1ps
module tb_dot32;
  reg clk=0; always #5 clk=~clk;
  reg rst=1,ce=1,valid_i=0;
  reg [1023:0] record_i;
  reg [1:0] mode=0;
  wire [2:0] valid_o;
  wire [31:0] y[0:2];
  wire signed [31:0] ints[0:1];
  ssnf_dot32 s(clk,rst,ce,valid_i&&(mode==0),record_i,valid_o[0],y[0],ints[0]);
  mxint8_dot32 m(clk,rst,ce,valid_i&&(mode==1),record_i,valid_o[1],y[1],ints[1]);
  bf16_dot32 b(clk,rst,ce,valid_i&&(mode==2),record_i,valid_o[2],y[2]);
  reg [31:0] expected[0:2][0:100000],expected_int[0:2][0:100000];
  integer accepted_cycle[0:2][0:100000];
  integer head[0:2],tail[0:2],latency[0:2];
  integer f,rc,kind,cycles=0,accepted=0,completed=0,i;
  reg [1023:0] rec;
  reg [31:0] exp_y,exp_int,noise=32'hc0ffee;
  string filename;
  reg finished=0;
  always @(posedge clk) begin
    if(!rst&&ce) begin
      cycles=cycles+1;
      if(valid_i) begin
        expected[mode][tail[mode]]=exp_y;
        expected_int[mode][tail[mode]]=exp_int;
        accepted_cycle[mode][tail[mode]]=cycles;
        tail[mode]=tail[mode]+1; accepted=accepted+1;
      end
      #1;
      for(i=0;i<3;i=i+1) if(valid_o[i]) begin
        if(head[i]>=tail[i]) $fatal(1,"Unexpected output mode=%0d",i);
        if(y[i]!==expected[i][head[i]])
          $fatal(1,"Dot mismatch mode=%0d index=%0d got=%h expected=%h",i,head[i],y[i],expected[i][head[i]]);
        if(i<2 && ints[i]!==expected_int[i][head[i]])
          $fatal(1,"Integer mismatch mode=%0d index=%0d got=%h expected=%h",i,head[i],ints[i],expected_int[i][head[i]]);
        if(latency[i]==0) latency[i]=cycles-accepted_cycle[i][head[i]]+1;
        if(latency[i]!=(cycles-accepted_cycle[i][head[i]]+1)) $fatal(1,"Variable enabled-clock latency");
        head[i]=head[i]+1; completed=completed+1;
      end
    end
  end
  initial begin
    for(integer k=0;k<3;k=k+1) begin head[k]=0;tail[k]=0;latency[k]=0;end
    if(!$value$plusargs("VECTORS=%s",filename)) $fatal(1,"VECTORS is required");
    f=$fopen(filename,"r"); if(!f) $fatal(1,"Cannot open vectors");
    repeat(3) @(negedge clk); rst=0;
    while(!$feof(f)) begin
      rc=$fscanf(f,"%d %h %h %h\n",kind,rec,exp_y,exp_int);
      if(rc==4) begin
        valid_i=0;
        noise={noise[30:0],noise[31]^noise[21]^noise[1]^noise[0]};
        if(noise[2:0]==0) begin ce=0; repeat(2) @(negedge clk); end
        ce=1;
        if(noise[5:3]==0) @(negedge clk);
        record_i=rec; mode=kind; valid_i=1;
        @(negedge clk); valid_i=0;
      end
    end
    ce=1; repeat(25) @(negedge clk);
    if(accepted!=completed) $fatal(1,"Lost outputs accepted=%0d completed=%0d",accepted,completed);
    for(integer k=0;k<3;k=k+1)
      $display("MODE %0d CASES %0d LATENCY_ENABLED_STAGES %0d",k,head[k],latency[k]);
    $display("PASS DOT32 accepted=%0d completed=%0d with bubbles and clock-enable stalls",accepted,completed);
    $finish;
  end
  initial begin #100000000; $fatal(1,"Simulation timeout"); end
endmodule
